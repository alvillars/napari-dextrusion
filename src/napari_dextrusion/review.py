"""Review of predicted events (self-training), without any GUI code.

Workflow: ``dextrusion detect`` writes one probability map per class; :func:`collect_candidates`
turns them back into scored events, :func:`sample_candidates` draws a random percentage of every
class, and the user gives each one a verdict (a class, "non-event" or "exclude") stored in a
:class:`ReviewLog`. :func:`apply_verdicts` then writes the corrected events into the training
folder: class files ``<movie><suffix>`` and ``<movie>_nothing.zip`` (hand-picked non-events, which
``dextrusion train`` samples as class 0 when ``--add-nothing`` is above 1).

Everything is expressed in the coordinates of the movie that was analysed. The crop shown for
review is the window the network sees (frames and pixels at the network's scale) mapped back to
those coordinates, see :func:`crop_spec`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from dextrusion.config import DeXConfig
from dextrusion.inference import scale_factors
from dextrusion.io import read_movie, read_rois
from dextrusion.label import clean_points, load_points, roi_path, save_points
from dextrusion.postprocess import get_events

NOTHING_SUFFIX = "_nothing.zip"
EXCLUDE = "exclude"
LOG_VERSION = 1


def class_label(suffix: str) -> str:
    """``'_cell_division.zip'`` -> ``'division'`` (display name of a class)."""
    name = suffix[:-4] if suffix.endswith(".zip") else suffix
    return name.removeprefix("_cell_").removeprefix("_") or "no event"


def class_suffix(name: str) -> str:
    """Inverse of :func:`class_label` for names typed by the user: ``'division'`` -> suffix."""
    return f"_cell_{name.strip()}.zip"


# ------------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    """A detected event: class index ``cat`` of the detecting network, position, probability."""

    cat: int
    t: int
    y: int
    x: int
    proba: float  # mean probability of the event volume, 0-255 as in the probability maps

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.cat, self.t, self.y, self.x)


def proba_map_path(folder: str | Path, movie_stem: str, suffix: str) -> Path:
    return Path(folder) / f"{movie_stem}{suffix[:-4]}_rawproba.tif"


def collect_candidates(folder: str | Path, movie_stem: str, config: DeXConfig, init_shape,
                       volume_threshold: float = 800, proba_threshold: float = 180,
                       disxy: int = 10, distime: int = 4) -> dict[int, list[Candidate]]:
    """Events of every class found in the ``*_rawproba.tif`` maps of a detection folder.

    The thresholds must be those used for the detection if the candidates are to match its ROI
    files; the maps are read again (rather than the ROI zips) because only they hold the scores.
    """
    out: dict[int, list[Candidate]] = {}
    for cat in range(1, config.ncat):
        path = proba_map_path(folder, movie_stem, config.catnames[cat])
        if not path.exists():
            continue
        events = get_events(read_movie(path), init_shape, volume_threshold, proba_threshold,
                            disxy, distime)
        out[cat] = [Candidate(cat, int(e.t), int(e.y), int(e.x), float(e.proba)) for e in events]
    return out


def find_detection_suffixes(folder: str | Path, movie_stem: str) -> list[str]:
    """Class suffixes (``'_cell_division.zip'``) of the ROI files ``<movie><suffix>`` of a folder,
    without backups and without the hand-picked non-events."""
    folder = Path(folder)
    found = {p.name[len(movie_stem):] for p in folder.glob(f"{movie_stem}_*.zip")
             if not p.name.endswith(".bak.zip")}
    return sorted(found - {NOTHING_SUFFIX})


def map_score(pm: np.ndarray, init_shape, t: int, y: int, x: int) -> float:
    """Peak of a probability map (0-255, at the rescaled size) around a position of the movie."""
    mt, my, mx = (round(c * m / n) for c, m, n in zip((t, y, x), pm.shape, init_shape))
    window = pm[max(0, mt - 1): mt + 2, max(0, my - 3): my + 4, max(0, mx - 3): mx + 4]
    return float(window.max()) if window.size else float("nan")


def collect_from_rois(folder: str | Path, movie_stem: str, suffixes: list[str],
                      init_shape) -> dict[int, list[Candidate]]:
    """Events of a detection folder read from its ROI files, for runs made outside the plugin.

    Class ``i + 1`` is ``suffixes[i]``. The score is the peak of the matching ``*_rawproba.tif``
    map next to the ROI file (a different quantity from the mean event probability of
    :func:`collect_candidates`), or NaN when there is no map. Positions outside the movie mean the
    detections belong to another movie and raise ``ValueError``.
    """
    folder = Path(folder)
    out: dict[int, list[Candidate]] = {}
    for cat, suffix in enumerate(suffixes, start=1):
        zip_path = roi_path(folder, movie_stem, suffix)
        if not zip_path.exists():
            continue
        points = read_rois(zip_path)
        bad = [p for p in points if not all(0 <= c < n for c, n in zip(p, init_shape))]
        if bad:
            raise ValueError(f"{zip_path.name}: {len(bad)} event(s) lie outside the open movie "
                             f"{tuple(init_shape)} (first: {bad[0]}); are these detections of "
                             "another movie?")
        mp = proba_map_path(folder, movie_stem, suffix)
        pm = read_movie(mp) if mp.exists() else None
        out[cat] = [Candidate(cat, int(t), int(y), int(x),
                              map_score(pm, init_shape, t, y, x) if pm is not None else float("nan"))
                    for t, y, x in points]
    return out


def external_config(model_dir: str | Path | None, suffixes: list[str], model_config) -> DeXConfig:
    """Window geometry and classes for detections made outside the plugin: the network's if a
    model folder is given (``model_config`` reads it), else the standard window; the classes of
    the model first, then any class found in the folder that it does not have."""
    cfg = model_config(model_dir) if model_dir else DeXConfig(catnames=[""], ncat=1)
    names = list(cfg.catnames)
    names += [s for s in suffixes if s not in names]
    return replace(cfg, catnames=names, ncat=len(names))


def sample_candidates(candidates: dict[int, list[Candidate]], percent: dict[int, float],
                      seed: int = 0) -> list[Candidate]:
    """Random ``percent`` % of the events of each class (at least one if the class has any and
    ``percent > 0``), in a random order that mixes the classes. Deterministic given ``seed``."""
    chosen: list[Candidate] = []
    for cat, events in sorted(candidates.items()):
        pct = float(percent.get(cat, 0))
        if not events or pct <= 0:
            continue
        n = min(len(events), max(1, math.ceil(len(events) * min(pct, 100) / 100)))
        idx = np.random.default_rng([seed, cat]).choice(len(events), size=n, replace=False)
        chosen += [events[int(i)] for i in sorted(idx)]
    order = np.random.default_rng([seed, 10_000]).permutation(len(chosen))
    return [chosen[int(i)] for i in order]


# ------------------------------------------------------------------------------------- crop
@dataclass(frozen=True)
class CropSpec:
    """Extent of the review crop in movie units: frames before / after the event frame
    (the window is ``frame - before .. frame + after - 1``) and half size in y and x."""

    before: int
    after: int
    half_y: int
    half_x: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.before + self.after, 2 * self.half_y + 1, 2 * self.half_x + 1)


def crop_spec(config: DeXConfig, cell_diameter: float, extrusion_duration: float) -> CropSpec:
    """The network window (``nframes`` x ``2*half_size+1`` px at the network's scale) expressed in
    pixels / frames of a movie with the given cell size and event duration. The zoom rule is the
    one used by ``detect`` and ``prepare`` (only applied when the scales differ by > 30 %)."""
    rxy, rz = scale_factors(config.cell_diameter, cell_diameter, config.extrusion_duration,
                            extrusion_duration)
    return CropSpec(
        before=max(1, round(config.nframes[0] / rz)), after=max(1, round(config.nframes[1] / rz)),
        half_y=max(1, round(config.half_size[0] / rxy)),
        half_x=max(1, round(config.half_size[1] / rxy)))


def extract_crop(movie: np.ndarray, spec: CropSpec, t: int, y: int, x: int) -> np.ndarray:
    """``(T, Y, X)`` crop of the movie around ``(t, y, x)``, zero-padded beyond the edges so the
    shape is always ``spec.shape`` and the event stays at index ``spec.before`` / the centre."""
    lo = np.array([t - spec.before, y - spec.half_y, x - spec.half_x])
    hi = lo + np.array(spec.shape)
    lo_c = np.maximum(lo, 0)
    hi_c = np.minimum(hi, movie.shape)
    crop = np.zeros(spec.shape, dtype=movie.dtype)
    if np.all(hi_c > lo_c):
        dst = tuple(slice(a, b) for a, b in zip(lo_c - lo, hi_c - lo))
        src = tuple(slice(a, b) for a, b in zip(lo_c, hi_c))
        crop[dst] = movie[src]
    return crop


# ----------------------------------------------------------------------------------- verdicts
class ReviewLog:
    """The queue of events to review and the verdicts given so far, saved after every change.

    A verdict is a class suffix (``'_cell_division.zip'``), :data:`NOTHING_SUFFIX` (not an event)
    or :data:`EXCLUDE` (ignore this event). ``applied`` records where :func:`apply_verdicts`
    wrote the event, so that changing a verdict later moves the point instead of duplicating it.
    """

    def __init__(self, path: str | Path, entries: list[dict] | None = None, meta: dict | None = None):
        self.path = Path(path)
        self.entries: list[dict] = entries or []
        self.meta: dict = meta or {}

    @classmethod
    def new(cls, path: str | Path, queue: list[Candidate], catnames: list[str], **meta) -> ReviewLog:
        entries = [{"cat": c.cat, "class": catnames[c.cat], "t": c.t, "y": c.y, "x": c.x,
                    "proba": None if math.isnan(c.proba) else round(c.proba, 2), "verdict": None, "applied": None} for c in queue]
        log = cls(path, entries, {"version": LOG_VERSION, **meta})
        log.save()
        return log

    @classmethod
    def load(cls, path: str | Path) -> ReviewLog:
        d = json.loads(Path(path).read_text())
        return cls(path, d["entries"], d["meta"])

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"meta": self.meta, "entries": self.entries}, indent=1) + "\n")
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self.entries)

    def set_verdict(self, i: int, verdict: str | None) -> None:
        self.entries[i]["verdict"] = verdict
        self.save()

    def first_pending(self) -> int | None:
        return next((i for i, e in enumerate(self.entries) if e["verdict"] is None), None)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            if e["verdict"] is not None:
                out[e["verdict"]] = out.get(e["verdict"], 0) + 1
        return out

    def key_set(self) -> set[tuple[int, int, int, int]]:
        return {(e["cat"], e["t"], e["y"], e["x"]) for e in self.entries}


def apply_verdicts(log: ReviewLog, folder: str | Path, movie_stem: str, shape) -> dict[str, int]:
    """Write the verdicts into the training folder and return the number of events per file.

    Idempotent: only events whose verdict changed since the last call are touched. A point added
    earlier is removed from its old file when its verdict changes (or is undone); points that
    were annotated by hand are never removed. Files are written with ``.bak.zip`` backups.
    """
    folder = Path(folder)
    points: dict[str, list[tuple[int, int, int]]] = {}

    def pts(suffix: str):
        if suffix not in points:
            points[suffix] = clean_points(load_points(roi_path(folder, movie_stem, suffix)), shape)
        return points[suffix]

    touched: set[str] = set()
    for e in log.entries:
        target = e["verdict"] if e["verdict"] not in (None, EXCLUDE) else None
        if e["applied"] == target:
            continue
        p = clean_points([[e["t"], e["y"], e["x"]]], shape)[0]
        if e["applied"] is not None:
            pts(e["applied"])[:] = [q for q in pts(e["applied"]) if q != p]
            touched.add(e["applied"])
        added = target is not None and p not in pts(target)
        if added:
            pts(target).append(p)
            touched.add(target)
        # a point that was already annotated by hand is not ours to remove later
        e["applied"] = target if added else None
    for suffix in touched:
        save_points(roi_path(folder, movie_stem, suffix), points[suffix], shape)
    log.save()
    return {s: len(points[s]) for s in sorted(touched)}
