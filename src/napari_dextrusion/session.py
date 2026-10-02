"""State shared by the tabs of the plugin: the movie, the project folder and the scale."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dextrusion.config import DeXConfig
from dextrusion.io import CONFIG_FILE, is_native_model, read_movie, resolve_models
from dextrusion.label import link_movie


def model_config(path: str | Path) -> DeXConfig:
    """Configuration (classes, window size, scale) of a DeXNet or of the first net of a folder of
    DeXNets. Legacy Keras nets are read from their ``config.cfg`` without converting them."""
    first = resolve_models(path)[0]
    if is_native_model(first):
        return DeXConfig.from_json(first / CONFIG_FILE)
    return DeXConfig.from_legacy_cfg(first / "config.cfg")


@dataclass
class Session:
    movie_path: Path | None = None
    movie: np.ndarray | None = None
    project: Path | None = None
    cell_diameter: float = 25.0
    extrusion_duration: float = 4.5
    listeners: list = field(default_factory=list, repr=False)
    reload_annotations: object = field(default=None, repr=False)  # set by the annotate tab

    # ------------------------------------------------------------------------------ movie
    def open_movie(self, path: str | Path, project: str | Path | None = None) -> np.ndarray:
        path = Path(path)
        img = read_movie(path)
        if img.ndim != 3:
            raise ValueError(f"{path.name}: expected a (T, Y, X) movie, got shape {img.shape}. "
                             "Project multichannel / 3D data to 2D first.")
        self.movie_path, self.movie = path, img
        self.project = Path(project) if project else path.parent / f"{path.stem}_dextrusion"
        self.project.mkdir(parents=True, exist_ok=True)
        link_movie(path, self.project)
        self.notify()
        return img

    @property
    def stem(self) -> str:
        return self.movie_path.stem if self.movie_path else ""

    def require_movie(self) -> None:
        if self.movie is None or self.project is None:
            raise RuntimeError("open a movie first")

    # --------------------------------------------------------------------------- folders
    def detect_dir(self, name: str) -> Path:
        return self.project / "detect" / name

    def detect_runs(self) -> list[Path]:
        root = self.project / "detect" if self.project else None
        return sorted(p for p in root.iterdir() if p.is_dir()) if root and root.is_dir() else []

    @property
    def prepared_dir(self) -> Path:
        return self.project / "prepared"

    def run_dir(self, name: str) -> Path:
        return self.project / "runs" / name

    def review_log_path(self, detect_name: str) -> Path:
        return self.project / f"{self.stem}_review_{detect_name}.json"

    # ------------------------------------------------------------------------- listeners
    def notify(self) -> None:
        for f in list(self.listeners):
            f()


DETECT_META = "plugin_detect.json"


def write_detect_meta(folder: Path, **meta) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / DETECT_META).write_text(json.dumps(meta, indent=1, default=str) + "\n")


def read_detect_meta(folder: Path) -> dict | None:
    f = Path(folder) / DETECT_META
    return json.loads(f.read_text()) if f.exists() else None
