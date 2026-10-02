import sys

import numpy as np
import pytest
import tifffile
from dextrusion.io import read_rois
from dextrusion.label import save_points
from napari.components import ViewerModel
from napari.layers import Image
from qtpy.QtCore import QEventLoop, QTimer

from conftest import EVENTS
from napari_dextrusion.commands import dextrusion_argv
from napari_dextrusion.curves import HistoryPlot, read_history
from napari_dextrusion.jobs import JobRunner
from napari_dextrusion.review import NOTHING_SUFFIX
from napari_dextrusion.session import Session, write_detect_meta
from napari_dextrusion.tab_inference import InferenceTab
from napari_dextrusion.tab_review import CROP_LAYER, EXCLUDE_KEY, NOTHING_KEY, ReviewTab
from napari_dextrusion.tab_train import TrainTab, build_catnames, link_extra_data
from napari_dextrusion.widget import DextrusionWidget

DEATH, DIV = "_cell_death.zip", "_cell_division.zip"


def wait(signal, timeout_s=240):
    loop = QEventLoop()
    got = []
    signal.connect(lambda *a: (got.append(a), loop.quit()))
    QTimer.singleShot(timeout_s * 1000, loop.quit)
    loop.exec()
    assert got, "timeout"
    return got[0]


def test_widget_builds_and_opens_a_movie(qapp, movie_file, no_dialogs):
    viewer = ViewerModel()
    w = DextrusionWidget(viewer)
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == [
        "Detect", "Annotate", "Train", "Review"]
    w.open_movie(movie_file)
    assert "movie" in viewer.layers and w.session.project.name == "mov_dextrusion"
    assert (w.session.project / "mov.tif").is_symlink()  # linked, never copied
    w.diameter.setValue(50)
    assert w.session.cell_diameter == 50
    w.open_movie(movie_file.parent / "missing.tif")  # reported to the user, no exception
    assert len(no_dialogs) == 1
    flat = movie_file.parent / "flat.tif"
    tifffile.imwrite(flat, np.zeros((8, 8), np.uint8))
    w.open_movie(flat)
    assert "(T, Y, X)" in no_dialogs[-1] and w.session.movie_path == movie_file


def test_reader_accepts_only_3d_tifs(tmp_path, movie_file):
    from napari_dextrusion.reader import napari_get_reader

    assert napari_get_reader(str(movie_file)) is not None
    flat = tmp_path / "flat.tif"
    tifffile.imwrite(flat, np.zeros((8, 8), np.uint8))
    assert napari_get_reader(str(flat)) is None
    assert napari_get_reader(str(tmp_path / "a.npy")) is None
    data, attrs, kind = napari_get_reader(str(movie_file))(str(movie_file))[0]
    assert data.shape[0] == 32 and attrs["name"] == "mov" and kind == "image"


def test_job_runner_streams_output_and_stops_on_failure(qapp):
    r = JobRunner()
    lines = []
    r.line.connect(lines.append)
    ok_cmd = dextrusion_argv("detect", "--help")
    r.run([[sys.executable, "-c", "print('hello'); print('world')"] * 1 + [], ok_cmd])
    (ok,) = wait(r.done)
    assert ok and "hello" in lines and any("--cell-diameter" in ln for ln in lines)
    lines.clear()
    r.run([[sys.executable, "-c", "import sys; sys.exit(3)"], [sys.executable, "-c", "print('no')"]])
    (ok,) = wait(r.done)
    assert not ok and "no" not in lines and any("exit code 3" in ln for ln in lines)
    r.run([[sys.executable, "-c", "import time; time.sleep(60)"]])
    QTimer.singleShot(300, r.stop)
    (ok,) = wait(r.done)
    assert not ok and "stopped" in lines and not r.running


def test_history_plot_reads_and_renders(qapp, tmp_path):
    f = tmp_path / "history.csv"
    assert read_history(f) == {}
    f.write_text("epoch,lr,loss,acc,val_loss,val_acc\n1,0.01,1.2,0.5,1.1,0.55\n2,0.01,0.9,0.7,nan,nan\n")
    h = read_history(f)
    assert h["epoch"] == [1.0, 2.0] and h["loss"] == [1.2, 0.9]
    plot = HistoryPlot()
    plot.resize(300, 300)
    plot.watch(f)
    img = plot.grab().toImage()
    pixels = {img.pixel(x, y) for x in range(300) for y in range(300)}
    assert 0xFF1F77B4 in pixels  # the training curve is drawn
    assert plot.data["epoch"] == [1.0, 2.0]
    plot.watch(None)


def test_build_catnames_and_link_extra_data(tmp_path):
    base = ["", DEATH, "_cell_sop.zip", DIV]
    assert build_catnames(base, "division, delamination") == base + ["_cell_delamination.zip"]
    assert build_catnames(None, "a, b") == ["", "_cell_a.zip", "_cell_b.zip"]
    with pytest.raises(ValueError):
        build_catnames(None, " ")
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for n in ("a.tif", "a_cell_death.zip", "a_cell_death.bak.zip", "notes.txt"):
        (src / n).write_bytes(b"x")
    assert link_extra_data(src, dst) == 2
    assert sorted(p.name for p in dst.iterdir()) == ["a.tif", "a_cell_death.zip"]
    assert all(p.is_symlink() for p in dst.iterdir())
    assert link_extra_data(src, dst) == 0  # nothing is overwritten


def make_detection(session, tiny_model, name="net"):
    """A fake detection folder (probability maps + settings) for the review tab."""
    folder = session.detect_dir(name)
    folder.mkdir(parents=True)
    for suffix, centres in ((DEATH, EVENTS[:2]), (DIV, EVENTS[2:])):
        pm = np.zeros(session.movie.shape, np.uint8)
        for t, y, x in centres:
            pm[t - 3 : t + 4, y - 10 : y + 10, x - 10 : x + 10] = 230
        tifffile.imwrite(folder / f"{session.stem}{suffix[:-4]}_rawproba.tif", pm)
    write_detect_meta(folder, models=tiny_model, volume_threshold=800, proba_threshold=180,
                      disxy=10, distime=4, cell_diameter=25, extrusion_duration=4.5)
    return folder


def test_review_flow(qapp, movie_file, tiny_model):
    viewer = ViewerModel()
    s = Session()
    s.open_movie(movie_file)
    folder = make_detection(s, tiny_model)
    tab = ReviewTab(viewer, s)
    assert tab.runs.currentText() == "net"
    tab.load_detections()
    assert sorted(tab.percent) == [1, 2, 3] and not tab.percent[2].isEnabled()  # no map for sop
    for w in tab.percent.values():
        w.setValue(100)
    tab.start()
    run = tab.run
    assert run.n == 4 and CROP_LAYER in viewer.layers
    crop = viewer.layers[CROP_LAYER]
    assert crop.data.shape == run.spec.shape == (10, 45, 45)
    first = run.entry
    assert tuple(crop.translate) == (first["t"] - 5, first["y"] - 22, first["x"] - 22)
    # 3 classes + not an event + exclude are bound on the crop layer (not on the viewer)
    assert len(crop.keymap) == len(tab.keys) == 5
    # verdicts: the first event is relabelled, the second is not an event, one is excluded
    cats = {e["class"] for e in run.log.entries}
    assert cats == {DEATH, DIV}
    tab._verdict(DIV)
    tab._verdict(NOTHING_SUFFIX)
    tab._verdict("exclude")
    assert not run.finished
    tab._verdict(DEATH)
    assert run.finished and "All events reviewed" in tab.info.text()
    n_div = len(read_rois(s.project / f"mov{DIV}"))
    assert n_div >= 1 and len(read_rois(s.project / f"mov{NOTHING_SUFFIX}")) == 1
    # going back and changing a verdict moves the point
    tab._move("back")
    entry = run.entry
    tab._verdict(NOTHING_SUFFIX)
    assert len(read_rois(s.project / f"mov{NOTHING_SUFFIX}")) in (1, 2)
    assert entry["applied"] == NOTHING_SUFFIX
    # a second tab on the same project resumes the saved log, or starts a new sample
    tab2 = ReviewTab(viewer, s)
    tab2.load_detections()
    tab2._ask_resume = lambda path: "resume"
    tab2.start()
    assert tab2.run.finished and tab2.run.log.counts() == run.log.counts()
    tab2._ask_resume = lambda path: "new"
    tab2.percent[1].setValue(50)
    tab2.start()
    assert not tab2.run.finished and (s.project / "mov_review_net.previous0.json").exists()
    assert folder.exists()


def test_review_keys_do_not_clash_with_napari_defaults():
    from dextrusion.label import CLASS_KEYS

    taken = {k.lower() for k in Image.class_keymap}
    ours = {*CLASS_KEYS, NOTHING_KEY, EXCLUDE_KEY}
    assert not ours & taken, ours & taken


def test_detect_then_train_through_the_real_cli(qapp, movie_file, tiny_model):
    """Detect with a tiny untrained net, annotate, then rescale + fine-tune with a new class."""
    viewer = ViewerModel()
    s = Session()
    s.open_movie(movie_file)
    det = InferenceTab(viewer, s)
    det.set_model(tiny_model)
    det.device.setCurrentText("cpu")
    det.dxy.setValue(40)
    det.dz.setValue(8)
    det.start()
    wait(det.runner.done)
    out = s.detect_runs()[0]
    assert (out / "plugin_detect.json").exists()
    msg = det.log.toPlainText()[-1500:] + str(sorted(p.name for p in out.iterdir()))
    assert list(out.glob("mov_cell_*_rawproba.tif")) and (out / "mov_cell_death.zip").exists(), msg
    assert "Detected" in det.info.text() and any(n.startswith("detected ") for n in
                                                  (layer.name for layer in viewer.layers))

    shape = s.movie.shape
    save_points(s.project / f"mov{DEATH}", [(t, y, x) for t, y, x in EVENTS[:2]], shape)
    save_points(s.project / "mov_cell_delamination.zip", [(t, y, x) for t, y, x in EVENTS[2:]], shape)
    save_points(s.project / f"mov{NOTHING_SUFFIX}", [(5, 20, 20), (28, 70, 20)], shape)
    trained = []
    tr = TrainTab(viewer, s, on_trained=trained.append)
    tr.set_init(tiny_model)
    tr.classes.setText("delamination")
    tr.epochs.setValue(2)
    tr.batch.setValue(8)
    tr.device.setCurrentText("cpu")
    tr.freeze.setChecked(True)
    tr.start()
    (ok,) = wait(tr.runner.done)
    assert ok, tr.log.toPlainText()[-800:]
    run = trained[0]
    h = read_history(run / "history.csv")
    assert h["epoch"] == [1.0, 2.0]
    from dextrusion.config import DeXConfig

    cfg = DeXConfig.from_json(run / "config.json")
    assert cfg.catnames[-1] == "_cell_delamination.zip" and cfg.ncat == 5
    assert (s.prepared_dir / "mov_nothing.zip").exists()  # hard negatives were carried over


def test_review_loads_roi_files_of_an_earlier_run(qapp, movie_file, tmp_path, no_dialogs):
    """Detections made outside the plugin: only the ROI zips (no plugin_detect.json, no maps)."""
    from dextrusion.io import create_roi, write_rois

    old = tmp_path / "earlier_run" / "ft"
    old.mkdir(parents=True)
    write_rois(old / f"mov{DEATH}", [create_roi(p, 1) for p in EVENTS[:2]], verbose=False)
    write_rois(old / "mov_cell_delamination.zip", [create_roi(p, 1) for p in EVENTS[2:]],
               verbose=False)
    write_rois(old / f"mov{DEATH[:-4]}.bak.zip", [create_roi((1, 1, 1), 1)], verbose=False)
    viewer = ViewerModel()
    s = Session()
    s.open_movie(movie_file)
    tab = ReviewTab(viewer, s)
    assert tab.runs.count() == 0  # nothing from the plugin yet
    tab.add_folder(old)
    assert tab.runs.currentText() == "earlier_run/ft"
    tab.load_detections()
    assert not no_dialogs and sorted(tab.percent) == [1, 2]
    for w in tab.percent.values():
        w.setValue(100)
    tab.start()
    run = tab.run
    assert run.n == 4 and run.log.entries[0]["proba"] is None
    assert run.log.meta["classes"] == [DEATH, "_cell_delamination.zip"]
    assert "n/a" in tab.info.text()
    assert s.review_log_path("earlier_run-ft").exists()
    tab._verdict("_cell_delamination.zip")
    assert read_rois(s.project / "mov_cell_delamination.zip")
    # the folder stays listed after the project's runs are refreshed
    tab.refresh_runs()
    assert tab.runs.count() == 1
    # detections of another movie are refused with a message, not a crash
    far = tmp_path / "far"
    far.mkdir()
    write_rois(far / f"mov{DEATH}", [create_roi((5, 900, 900), 1)], verbose=False)
    tab.add_folder(far)
    tab.load_detections()
    assert any("outside the open movie" in m for m in no_dialogs)
