"""Detection tab: run ``dextrusion detect`` with the chosen model(s) on the open movie."""

from __future__ import annotations

import logging
from pathlib import Path

from dextrusion.label import PALETTE, load_points, point_style_kwargs, roi_path
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .commands import DetectParams, detect_args, dextrusion_argv
from .jobs import JobRunner
from .review import class_label
from .session import Session, model_config, write_detect_meta
from .ui import LogView, PathRow, float_spin, int_spin

log = logging.getLogger("napari_dextrusion")


def unused_name(parent: Path, name: str) -> str:
    """``name``, or ``name_2``, ``name_3``... so that a previous detection is never overwritten."""
    candidate, i = name, 1
    while (parent / candidate).exists():
        i += 1
        candidate = f"{name}_{i}"
    return candidate


class InferenceTab(QWidget):
    def __init__(self, viewer, session: Session):
        super().__init__()
        self.viewer, self.session = viewer, session
        self.runner = JobRunner(self)
        self.runner.line.connect(lambda s: self.log.add(s))
        self.runner.done.connect(self._done)
        self._current: tuple[str, object] | None = None

        self.models = PathRow(caption="DeXNet folder (or a folder of DeXNets = ensemble)")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        self.dxy = int_spin(10, 1, 100, "spatial step of the sliding window (pixels)")
        self.dz = int_spin(2, 1, 20, "temporal step of the sliding window (frames)")
        self.volume = float_spin(800, 0, 1e6, 50, 0, "minimum event volume in the probability map")
        self.proba = float_spin(180, 0, 255, 5, 0, "minimum mean probability of an event (0-255)")
        self.disxy = int_spin(10, 1, 200, "events closer than this (pixels) are merged")
        self.distime = int_spin(4, 1, 50, "events closer than this (frames) are merged")
        self.run_btn, self.stop_btn = QPushButton("Detect events"), QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.runner.stop)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.log = LogView()

        form = QFormLayout()
        form.addRow("model(s)", self.models)
        form.addRow("device", self.device)
        form.addRow("window step xy / t", self._pair(self.dxy, self.dz))
        form.addRow("volume / probability threshold", self._pair(self.volume, self.proba))
        form.addRow("merge xy / t", self._pair(self.disxy, self.distime))
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.info)
        lay.addWidget(self.run_btn)
        lay.addWidget(self.stop_btn)
        lay.addWidget(self.log, 1)

    @staticmethod
    def _pair(a, b) -> QWidget:
        from qtpy.QtWidgets import QHBoxLayout

        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    def set_model(self, path: str | Path) -> None:
        self.models.setText(str(path))

    def params(self, outdir: Path) -> DetectParams:
        s = self.session
        dev = self.device.currentText()
        return DetectParams(
            Path(self.models.text()), outdir, s.cell_diameter, s.extrusion_duration,
            self.dxy.value(), self.dz.value(), self.volume.value(), self.proba.value(),
            self.disxy.value(), self.distime.value(), None if dev == "auto" else dev)

    def start(self) -> None:
        try:
            self.session.require_movie()
            if not self.models.text():
                raise RuntimeError("choose a model folder")
            cfg = model_config(self.models.text())
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Detect events", str(e))
            return
        name = unused_name(self.session.project / "detect", Path(self.models.text()).name)
        out = self.session.detect_dir(name)
        p = self.params(out)
        out.mkdir(parents=True, exist_ok=True)
        write_detect_meta(
            out, models=p.models, catnames=cfg.catnames, cell_diameter=p.cell_diameter,
            extrusion_duration=p.extrusion_duration, volume_threshold=p.volume_threshold,
            proba_threshold=p.proba_threshold, disxy=p.disxy, distime=p.distime,
            movie=self.session.movie_path, shape=list(self.session.movie.shape))
        self._current = (name, cfg)
        self.log.add(f"Detection results go to {out}")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.runner.run([dextrusion_argv(*detect_args(self.session.movie_path, p))])

    def _done(self, ok: bool) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if not ok or self._current is None:
            self.info.setText("Detection did not finish, see the log.")
            return
        name, cfg = self._current
        folder = self.session.detect_dir(name)
        shown = []
        for i, suffix in enumerate(cfg.catnames[1:]):
            pts = load_points(roi_path(folder, self.session.stem, suffix))
            self.viewer.add_points(
                pts, ndim=3, name=f"detected {class_label(suffix)} ({name})",
                size=self.session.cell_diameter, face_color="transparent",
                **point_style_kwargs(PALETTE[i % len(PALETTE)]))
            shown.append(f"{class_label(suffix)}: {len(pts)}")
        self.info.setText("Detected " + ", ".join(shown))
        self.session.notify()
