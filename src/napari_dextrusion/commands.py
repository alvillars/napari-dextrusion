"""Command lines for the ``dextrusion`` steps run by the plugin (GUI-free, testable).

The plugin runs detection, rescaling and training as separate processes through the public
``dextrusion`` command line: the GUI never blocks, CUDA memory is released when a step ends and
the plugin only depends on the documented options.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_ENTRY = "import sys; from dextrusion.cli import main; sys.exit(main(sys.argv[1:]))"


def dextrusion_argv(*args: str | Path) -> list[str]:
    """``python -c <dextrusion main> args...`` with the interpreter that runs the plugin."""
    return [sys.executable, "-c", _ENTRY, *map(str, args)]


@dataclass
class DetectParams:
    models: Path
    outdir: Path
    cell_diameter: float = 25
    extrusion_duration: float = 4.5
    dxy: int = 10
    dz: int = 2
    volume_threshold: float = 800
    proba_threshold: float = 180
    disxy: int = 10
    distime: int = 4
    device: str | None = None  # None: let dextrusion choose (cuda if available)


def detect_args(movie: str | Path, p: DetectParams) -> list[str]:
    a = ["detect", movie, "-m", p.models, "-o", p.outdir, "--cell-diameter", p.cell_diameter,
         "--extrusion-duration", p.extrusion_duration, "--dxy", p.dxy, "--dz", p.dz,
         "--volume-threshold", p.volume_threshold, "--proba-threshold", p.proba_threshold,
         "--disxy", p.disxy, "--distime", p.distime]
    if p.device:
        a += ["--device", p.device]
    return [str(v) for v in a]


def prepare_args(movie: str | Path, out: str | Path, rois_dir: str | Path, cell_diameter: float,
                 extrusion_duration: float) -> list[str]:
    return [str(v) for v in ["prepare", movie, "-o", out, "--cell-diameter", cell_diameter,
                             "--extrusion-duration", extrusion_duration, "--rois-dir", rois_dir]]


@dataclass
class TrainParams:
    data: Path  # folder with the (prepared) movie and its ROI files
    out: Path
    init_from: Path | None = None
    catnames: list[str] | None = None  # ROI suffixes, first is ''; with init_from: old + new
    epochs: int = 10
    lr: float = 0.01
    naug: int = 3
    add_nothing: int = 10
    val_ratio: float = 0.2
    batch_size: int = 30
    freeze_cnn: bool = False
    oversample: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    workers: int = 0
    device: str | None = None


def train_args(p: TrainParams) -> list[str]:
    a: list = ["train", p.data, "-o", p.out, "--epochs", p.epochs, "--lr", p.lr, "--naug", p.naug,
               "--add-nothing", p.add_nothing, "--val-ratio", p.val_ratio, "--batch-size", p.batch_size,
         "--seed", p.seed,
               "--workers", p.workers]
    if p.init_from:
        a += ["--init-from", p.init_from]
    if p.catnames:
        a += ["--catnames", *p.catnames]
    if p.freeze_cnn:
        a.append("--freeze-cnn")
    if p.oversample:
        a += ["--oversample", *(f"{k}={v}" for k, v in p.oversample.items())]
    if p.device:
        a += ["--device", p.device]
    return [str(v) for v in a]
