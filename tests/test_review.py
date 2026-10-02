import numpy as np
import tifffile
from dextrusion.config import DeXConfig
from dextrusion.io import read_rois

from napari_dextrusion.review import (
    EXCLUDE,
    NOTHING_SUFFIX,
    Candidate,
    CropSpec,
    ReviewLog,
    apply_verdicts,
    class_label,
    class_suffix,
    collect_candidates,
    crop_spec,
    extract_crop,
    sample_candidates,
)

CFG = DeXConfig()  # 4 classes, 5+5 frames, 45 x 45 px, cell 25 px, 4.5 frames
DIV = "_cell_division.zip"
DEATH = "_cell_death.zip"
SHAPE = (40, 120, 140)


def cands(n, cat):
    return [Candidate(cat, 5 + i, 30 + i, 40 + i, 200.0 + i) for i in range(n)]


def test_class_names_roundtrip():
    assert class_label("_cell_division.zip") == "division"
    assert class_label("_nothing.zip") == "nothing"
    assert class_label("") == "no event"
    assert class_suffix("division") == DIV


def test_sampling_percent_per_class_is_seeded_and_mixed():
    c = {1: cands(20, 1), 3: cands(10, 3), 2: cands(5, 2)}
    s = sample_candidates(c, {1: 25, 3: 50, 2: 0}, seed=4)
    by = {k: [e for e in s if e.cat == k] for k in (1, 2, 3)}
    assert (len(by[1]), len(by[2]), len(by[3])) == (5, 0, 5)
    assert len(set(s)) == len(s)  # no event twice
    assert s == sample_candidates(c, {1: 25, 3: 50, 2: 0}, seed=4)
    assert s != sample_candidates(c, {1: 25, 3: 50, 2: 0}, seed=5)
    assert [e.cat for e in s] != sorted(e.cat for e in s)  # classes are interleaved


def test_sampling_small_classes_and_extremes():
    c = {1: cands(3, 1), 3: []}
    assert len(sample_candidates(c, {1: 1, 3: 100})) == 1  # at least one when percent > 0
    assert len(sample_candidates(c, {1: 100})) == 3
    assert len(sample_candidates(c, {1: 250})) == 3  # capped
    assert sample_candidates(c, {}) == []


def test_crop_spec_follows_the_network_scale():
    assert crop_spec(CFG, 25, 4.5) == CropSpec(5, 5, 22, 22)  # same scale: the window itself
    assert crop_spec(CFG, 50, 4.5) == CropSpec(5, 5, 44, 44)  # cells twice as big: 90 px box
    assert crop_spec(CFG, 30, 4.5) == CropSpec(5, 5, 22, 22)  # within 30 %: no zoom, like detect
    s = crop_spec(CFG, 25, 9)  # events twice as slow: window spans twice the frames
    assert (s.before, s.after) == (10, 10)
    assert crop_spec(CFG, 50, 4.5).shape == (10, 89, 89)


def test_extract_crop_inside_and_padded_at_the_edges():
    movie = np.arange(np.prod(SHAPE), dtype=np.uint16).reshape(SHAPE)
    spec = CropSpec(5, 5, 22, 22)
    crop = extract_crop(movie, spec, 20, 60, 70)
    assert crop.shape == spec.shape
    np.testing.assert_array_equal(crop, movie[15:25, 38:83, 48:93])
    assert crop[spec.before, 22, 22] == movie[20, 60, 70]  # event at the centre
    for t, y, x in [(0, 0, 0), (SHAPE[0] - 1, SHAPE[1] - 1, SHAPE[2] - 1), (2, 10, 135)]:
        c = extract_crop(movie, spec, t, y, x)
        assert c.shape == spec.shape and c.dtype == movie.dtype
        assert c[spec.before, 22, 22] == movie[t, y, x]  # still centred
    assert extract_crop(movie, spec, 2, 5, 5)[0, 0, 0] == 0  # padding with zeros
    far = extract_crop(movie, spec, 500, 500, 500)  # fully outside: all zeros, no crash
    assert far.shape == spec.shape and not far.any()


def test_collect_candidates_reads_the_probability_maps(tmp_path):
    pm = np.zeros((20, 60, 70), np.uint8)
    pm[6:14, 20:40, 25:45] = 230  # one blob, big enough for the default volume threshold
    tifffile.imwrite(tmp_path / f"m{DEATH[:-4]}_rawproba.tif", pm)
    out = collect_candidates(tmp_path, "m", CFG, (20, 60, 70))
    assert list(out) == [1]  # the other classes have no map
    (c,) = out[1]
    assert 6 <= c.t <= 13 and 20 <= c.y <= 40 and 25 <= c.x <= 45 and c.proba > 180


def make_log(tmp_path, queue):
    return ReviewLog.new(tmp_path / "m_review.json", queue, CFG.catnames, movie="m", seed=1)


def test_log_roundtrip_and_resume(tmp_path):
    q = cands(3, 1)
    log = make_log(tmp_path, q)
    assert len(log) == 3 and log.first_pending() == 0 and log.meta["seed"] == 1
    log.set_verdict(0, DIV)
    log.set_verdict(1, EXCLUDE)
    again = ReviewLog.load(tmp_path / "m_review.json")
    assert again.first_pending() == 2
    assert again.counts() == {DIV: 1, EXCLUDE: 1}
    assert again.key_set() == {c.key for c in q}
    assert again.entries[0]["class"] == DEATH


def test_apply_writes_class_files_and_nothing_file(tmp_path):
    q = cands(4, 1)
    log = make_log(tmp_path, q)
    log.set_verdict(0, DIV)  # predicted extrusion, really a division
    log.set_verdict(1, DEATH)  # confirmed
    log.set_verdict(2, NOTHING_SUFFIX)
    log.set_verdict(3, EXCLUDE)
    n = apply_verdicts(log, tmp_path, "m", SHAPE)
    assert n == {DEATH: 1, DIV: 1, NOTHING_SUFFIX: 1}
    assert read_rois(tmp_path / f"m{DIV}") == [(5, 30, 40)]
    assert read_rois(tmp_path / f"m{DEATH}") == [(6, 31, 41)]
    assert read_rois(tmp_path / f"m{NOTHING_SUFFIX}") == [(7, 32, 42)]
    assert apply_verdicts(log, tmp_path, "m", SHAPE) == {}  # idempotent: nothing changed


def test_changing_or_undoing_a_verdict_moves_the_point(tmp_path):
    log = make_log(tmp_path, cands(1, 1))
    log.set_verdict(0, DIV)
    apply_verdicts(log, tmp_path, "m", SHAPE)
    log.set_verdict(0, NOTHING_SUFFIX)
    apply_verdicts(log, tmp_path, "m", SHAPE)
    assert read_rois(tmp_path / f"m{DIV}") == []  # valid empty file, point moved
    assert read_rois(tmp_path / f"m{NOTHING_SUFFIX}") == [(5, 30, 40)]
    log.set_verdict(0, None)  # undo
    apply_verdicts(log, tmp_path, "m", SHAPE)
    assert read_rois(tmp_path / f"m{NOTHING_SUFFIX}") == []


def test_hand_annotated_points_are_never_removed(tmp_path):
    from dextrusion.label import save_points

    save_points(tmp_path / f"m{DIV}", [[5, 30, 40], [9, 90, 90]], SHAPE)
    log = make_log(tmp_path, cands(1, 1))
    log.set_verdict(0, DIV)  # same place as the manual point
    apply_verdicts(log, tmp_path, "m", SHAPE)
    log.set_verdict(0, EXCLUDE)
    apply_verdicts(log, tmp_path, "m", SHAPE)
    assert read_rois(tmp_path / f"m{DIV}") == [(5, 30, 40), (9, 90, 90)]


def test_apply_keeps_existing_annotations_and_clips(tmp_path):
    from dextrusion.label import save_points

    save_points(tmp_path / f"m{DIV}", [[9, 90, 90]], SHAPE)
    log = make_log(tmp_path, [Candidate(1, 3, 10, 10, 200.0)])
    log.set_verdict(0, DIV)
    apply_verdicts(log, tmp_path, "m", SHAPE)
    assert read_rois(tmp_path / f"m{DIV}") == [(3, 10, 10), (9, 90, 90)]
    assert (tmp_path / "m_cell_division.bak.zip").exists()


def test_empty_queue(tmp_path):
    log = make_log(tmp_path, [])
    assert log.first_pending() is None and log.counts() == {}
    assert apply_verdicts(log, tmp_path, "m", SHAPE) == {}
