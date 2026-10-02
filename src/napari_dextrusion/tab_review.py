"""Review tab: look at a sample of predicted events and correct them (self-training)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from dextrusion.label import CLASS_KEYS, PALETTE, point_style_kwargs
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .review import (
    EXCLUDE,
    NOTHING_SUFFIX,
    ReviewLog,
    class_label,
    collect_candidates,
    collect_from_rois,
    crop_spec,
    external_config,
    find_detection_suffixes,
    sample_candidates,
)
from .review_run import ReviewRun
from .session import Session, model_config, read_detect_meta
from .ui import PathRow, float_spin, int_spin

log = logging.getLogger("napari_dextrusion")
NOTHING_KEY, EXCLUDE_KEY = "n", "u"
CROP_LAYER, MARK_LAYER = "review crop", "review event"


class ReviewTab(QWidget):
    def __init__(self, viewer, session: Session, goto_train=None):
        super().__init__()
        self.viewer, self.session, self.goto_train = viewer, session, goto_train
        self.run: ReviewRun | None = None
        self.candidates: dict = {}
        self.cfg = None
        self.meta: dict | None = None
        self.detect_name = ""
        self.crop_layer = self.mark_layer = None
        self.limits: tuple[float, float] | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._frame = 0

        self.runs = QComboBox()
        self.browse_btn = QPushButton("Other detection folder…")
        self.browse_btn.setToolTip("a folder with <movie>_cell_*.zip ROI files (and, optionally, "
                                   "the *_rawproba.tif maps) from an earlier run")
        self.browse_btn.clicked.connect(self.browse_folder)
        self.models_row = PathRow(caption="DeXNet used for the detections (window size, classes)")
        self.models_row.setToolTip("only for detections made outside the plugin; empty = standard "
                                   "window (10 frames, 45 px, 25 px cells, 4.5 frames)")
        self.load_btn = QPushButton("Load detections")
        self.load_btn.clicked.connect(self.load_detections)
        self.percent_form = QFormLayout()
        self.percent: dict[int, object] = {}
        self.seed = int_spin(0, 0, 10**6, "seed of the random sample")
        self.start_btn = QPushButton("Start / resume review")
        self.start_btn.clicked.connect(self.start)
        self.start_btn.setEnabled(False)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.verdict_grid = QGridLayout()
        self.verdict_box = QWidget()
        self.verdict_box.setLayout(self.verdict_grid)
        self.back_btn, self.skip_btn = QPushButton("◀ Back"), QPushButton("Skip ▶")
        self.back_btn.clicked.connect(lambda: self._move("back"))
        self.skip_btn.clicked.connect(lambda: self._move("skip"))
        self.loop = QCheckBox("play the crop in a loop")
        self.loop.setChecked(True)
        self.loop.toggled.connect(self._loop_toggled)
        self.speed = float_spin(8, 1, 30, 1, 0, "frames per second")
        self.speed.valueChanged.connect(lambda _: self._loop_toggled(self.loop.isChecked()))
        self.retrain_btn = QPushButton("Retrain with these annotations →")
        self.retrain_btn.clicked.connect(lambda: self.goto_train and self.goto_train())

        lay = QVBoxLayout(self)
        top = QFormLayout()
        top.addRow("detections", self.runs)
        top.addRow("model (other folders)", self.models_row)
        lay.addLayout(top)
        lay.addWidget(self.browse_btn)
        lay.addWidget(self.load_btn)
        lay.addWidget(QLabel("% of the events of each class to review"))
        lay.addLayout(self.percent_form)
        seed_form = QFormLayout()
        seed_form.addRow("seed", self.seed)
        lay.addLayout(seed_form)
        lay.addWidget(self.start_btn)
        lay.addWidget(self.info)
        lay.addWidget(self.verdict_box)
        nav = QGridLayout()
        nav.addWidget(self.back_btn, 0, 0)
        nav.addWidget(self.skip_btn, 0, 1)
        lay.addLayout(nav)
        lay.addWidget(self.loop)
        sp = QFormLayout()
        sp.addRow("speed (fps)", self.speed)
        lay.addLayout(sp)
        lay.addWidget(self.retrain_btn)
        lay.addStretch(1)
        self.verdict_box.setEnabled(False)
        session.listeners.append(self.refresh_runs)
        self.refresh_runs()

    # ----------------------------------------------------------------------- detections
    def refresh_runs(self) -> None:
        current = self.runs.currentData()
        extra = [(self.runs.itemText(i), self.runs.itemData(i)) for i in range(self.runs.count())
                 if not self._in_project(Path(self.runs.itemData(i)))]
        self.runs.clear()
        for p in self.session.detect_runs():
            self.runs.addItem(p.name, str(p))
        for name, path in extra:  # folders added with "Other detection folder…" stay listed
            self.runs.addItem(name, path)
        index = self.runs.findData(current)
        self.runs.setCurrentIndex(index if index >= 0 else self.runs.count() - 1)

    def _in_project(self, folder: Path) -> bool:
        return self.session.project is not None and folder.parent == self.session.project / "detect"

    def browse_folder(self) -> None:
        start = str(self.session.movie_path.parent) if self.session.movie_path else ""
        path = QFileDialog.getExistingDirectory(self, "Folder with detections", start)
        if path:
            self.add_folder(Path(path))

    def add_folder(self, folder: Path) -> None:
        """List a detection folder made outside the plugin and select it."""
        folder = Path(folder)
        index = self.runs.findData(str(folder))
        if index < 0:
            self.runs.addItem(f"{folder.parent.name}/{folder.name}", str(folder))
            index = self.runs.count() - 1
        self.runs.setCurrentIndex(index)

    def load_detections(self) -> None:
        s = self.session
        try:
            s.require_movie()
            data = self.runs.currentData()
            if not data:
                raise RuntimeError("no detection yet: run the Detect tab, or load a folder with "
                                   "'Other detection folder…'")
            folder = Path(data)
            meta = read_detect_meta(folder)
            if meta is not None:  # made by the Detect tab: settings were recorded
                name = folder.name
                cfg = model_config(meta["models"])
                cands = collect_candidates(
                    folder, s.stem, cfg, s.movie.shape, meta["volume_threshold"],
                    meta["proba_threshold"], meta["disxy"], meta["distime"])
            else:  # an earlier run: read the ROI files, scores from the maps if present
                name = folder.name if self._in_project(folder) else f"{folder.parent.name}-{folder.name}"
                suffixes = find_detection_suffixes(folder, s.stem)
                if not suffixes:
                    raise RuntimeError(f"no {s.stem}_*.zip ROI file in {folder}")
                cfg = external_config(self.models_row.path(), suffixes, model_config)
                cands = collect_from_rois(folder, s.stem, cfg.catnames[1:], s.movie.shape)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Review", str(e))
            return
        self.cfg, self.meta, self.candidates, self.detect_name = cfg, meta, cands, name
        while self.percent_form.rowCount():
            self.percent_form.removeRow(0)
        self.percent = {}
        for cat in range(1, cfg.ncat):
            n = len(cands.get(cat, []))
            w = float_spin(20, 0, 100, 5, 0, "random share of this class to review")
            w.setEnabled(n > 0)
            self.percent[cat] = w
            self.percent_form.addRow(f"{class_label(cfg.catnames[cat])} ({n} events)", w)
        self.start_btn.setEnabled(sum(map(len, cands.values())) > 0)
        self.info.setText("" if self.start_btn.isEnabled() else "No event was detected.")

    # ------------------------------------------------------------------------- review
    def start(self) -> None:
        s = self.session
        path = s.review_log_path(self.detect_name)
        try:
            if path.exists():
                choice = self._ask_resume(path)
                if choice == "resume":
                    rlog = ReviewLog.load(path)
                elif choice == "new":
                    old = path.with_name(f"{path.stem}.previous{len(self._previous(path))}.json")
                    path.rename(old)  # the verdicts already applied stay in the training files
                    rlog = self._new_log(path)
                else:
                    return
            else:
                rlog = self._new_log(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Review", str(e))
            return
        spec = crop_spec(self.cfg, s.cell_diameter, s.extrusion_duration)
        self.run = ReviewRun(rlog, s.movie, spec, s.project, s.stem,
                             on_applied=lambda: s.reload_annotations and s.reload_annotations())
        self._build_verdict_buttons(rlog.meta["classes"])
        self.verdict_box.setEnabled(bool(len(rlog)))
        flat = s.movie[:: max(1, len(s.movie) // 10)]
        self.limits = tuple(float(v) for v in np.percentile(flat, (1, 99.8)))
        self.show()

    @staticmethod
    def _previous(path: Path) -> list[Path]:
        return sorted(path.parent.glob(f"{path.stem}.previous*.json"))

    def _ask_resume(self, path: Path) -> str | None:
        """'resume', 'new' (draw a new sample, keep the old log) or None (cancel)."""
        box = QMessageBox(self)
        box.setWindowTitle("Review")
        box.setText(f"A review of these detections already exists ({path.name}).")
        resume = box.addButton("Resume it", QMessageBox.ButtonRole.AcceptRole)
        fresh = box.addButton("New sample (keep the old log)", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return "resume" if box.clickedButton() is resume else (
            "new" if box.clickedButton() is fresh else None)

    def _new_log(self, path: Path) -> ReviewLog:
        pct = {c: w.value() for c, w in self.percent.items()}
        queue = sample_candidates(self.candidates, pct, self.seed.value())
        if not queue:
            raise RuntimeError("the sample is empty: raise a percentage")
        classes = list(self.cfg.catnames[1:])
        return ReviewLog.new(path, queue, self.cfg.catnames, movie=self.session.stem,
                             detect=self.detect_name, seed=self.seed.value(), percent=pct,
                             classes=classes)

    def _build_verdict_buttons(self, classes: list[str]) -> None:
        while self.verdict_grid.count():
            self.verdict_grid.takeAt(0).widget().deleteLater()
        specs = [(f"{class_label(c)}  [{k}]", c, k) for c, k in zip(classes, CLASS_KEYS)]
        specs += [(f"not an event  [{NOTHING_KEY}]", NOTHING_SUFFIX, NOTHING_KEY),
                  (f"exclude  [{EXCLUDE_KEY}]", EXCLUDE, EXCLUDE_KEY)]
        self.keys = {}
        for i, (text, verdict, key) in enumerate(specs):
            b = QPushButton(text)
            b.clicked.connect(lambda _=None, v=verdict: self._verdict(v))
            self.verdict_grid.addWidget(b, i // 2, i % 2)
            self.keys[key] = verdict

    def _bind_keys(self, layer) -> None:
        for key, verdict in self.keys.items():
            layer.bind_key(key, lambda _layer, v=verdict: self._verdict(v), overwrite=True)

    def _verdict(self, verdict: str) -> None:
        if self.run is None:
            return
        self.run.verdict(verdict)
        self.show()

    def _move(self, how: str) -> None:
        if self.run is not None:
            getattr(self.run, how)()
            self.show()

    def show(self) -> None:
        """Display the current event: the network window cropped from the movie, in place."""
        run = self.run
        if run is None or not run.n:
            return
        e = run.entry
        crop = run.crop()
        t0, y0, x0 = run.origin()
        v = self.viewer
        if "movie" in v.layers:
            v.layers["movie"].visible = False
        if self.crop_layer is None or self.crop_layer not in v.layers:
            self.crop_layer = v.add_image(crop, name=CROP_LAYER, colormap="gray",
                                          contrast_limits=self.limits)
            self._bind_keys(self.crop_layer)
        self.crop_layer.data = crop
        self.crop_layer.translate = (t0, y0, x0)
        self.crop_layer.contrast_limits = self.limits
        marks = np.array([[t0 + k, e["y"], e["x"]] for k in range(crop.shape[0])], dtype=float)
        if self.mark_layer is None or self.mark_layer not in v.layers:
            self.mark_layer = v.add_points(
                marks, ndim=3, name=MARK_LAYER, size=self.session.cell_diameter,
                face_color="transparent", **point_style_kwargs(PALETTE[0], 0.08))
            self.mark_layer.editable = False
        self.mark_layer.data = marks
        v.layers.selection.active = self.crop_layer
        h, w = crop.shape[1:]
        v.camera.center = (0, e["y"], e["x"])
        v.camera.zoom = 520 / max(h, w)
        self._frame = 0
        self._tick()
        self._loop_toggled(self.loop.isChecked())
        mine = class_label(e["class"])
        given = e["verdict"]
        shown = "none yet" if given is None else (
            "excluded" if given == EXCLUDE else
            "not an event" if given == NOTHING_SUFFIX else class_label(given))
        self.info.setText(
            f"{run.progress()}\nevent {e['t']}, y {e['y']}, x {e['x']}\n"
            f"predicted: {mine} (score {'n/a' if e['proba'] is None else format(e['proba'], '.0f')})\nyour verdict: {shown}"
            + ("\nAll events reviewed." if run.finished else ""))

    def _loop_toggled(self, on: bool) -> None:
        if on and self.run is not None:
            self.timer.start(int(1000 / self.speed.value()))
        else:
            self.timer.stop()

    def _tick(self) -> None:
        run = self.run
        if run is None or run.n == 0:
            return
        t0 = run.origin()[0]
        n = run.spec.shape[0]
        lo, hi = max(0, t0), min(run.movie.shape[0] - 1, t0 + n - 1)
        self.viewer.dims.set_current_step(0, lo + self._frame % (hi - lo + 1))
        self._frame += 1
