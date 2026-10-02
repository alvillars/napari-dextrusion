"""Training tab: rescale the annotated movie, train / fine-tune, and plot the curves live."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .commands import TrainParams, dextrusion_argv, prepare_args, train_args
from .curves import HistoryPlot
from .jobs import JobRunner
from .review import class_suffix
from .session import Session, model_config
from .tab_inference import unused_name
from .ui import LogView, PathRow, float_spin, int_spin

log = logging.getLogger("napari_dextrusion")


def link_extra_data(extra: Path, dest: Path) -> int:
    """Symlink every ``.tif`` / ``.zip`` of ``extra`` into ``dest`` (existing names are kept).

    Used to train on the movie of the project together with another dataset (for instance the
    original one) without copying data. Returns the number of links made.
    """
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(extra.iterdir()):
        if f.suffix in (".tif", ".zip") and f.is_file() and not f.name.endswith(".bak.zip"):
            link = dest / f.name
            if not (link.exists() or link.is_symlink()):
                os.symlink(f.resolve(), link)
                n += 1
    return n


def build_catnames(base: list[str] | None, typed: str) -> list[str]:
    """Class suffixes for ``--catnames``: the classes of the network being fine-tuned (if any),
    then the typed class names that are not in it yet."""
    names = [n.strip() for n in typed.split(",") if n.strip()]
    out = list(base) if base else [""]
    for n in names:
        if class_suffix(n) not in out:
            out.append(class_suffix(n))
    if len(out) < 2:
        raise ValueError("give at least one class name")
    return out


class TrainTab(QWidget):
    def __init__(self, viewer, session: Session, on_trained=None):
        super().__init__()
        self.viewer, self.session, self.on_trained = viewer, session, on_trained
        self.runner = JobRunner(self)
        self.runner.line.connect(lambda s: self.log.add(s))
        self.runner.done.connect(self._done)
        self._run: Path | None = None

        self.init = PathRow(caption="DeXNet to fine-tune (leave empty to train from scratch)")
        self.classes = QLineEdit("division, delamination")
        self.classes.setToolTip("fine-tuning: classes to add to the network's; from scratch: all classes")
        self.extra = PathRow(caption="Other training data (folder with movies + ROI files)")
        self.oversample = int_spin(1, 1, 500, "sample the project movie this many times")
        self.epochs = int_spin(10, 1, 1000)
        self.lr = float_spin(0.01, 1e-5, 1.0, 0.005, 5)
        self.naug = int_spin(3, 1, 20, "augmentation factor (1: none)")
        self.add_nothing = int_spin(10, 0, 100, "copies of every hand-picked non-event; "
                                    "must be above 1 for the non-events to be used")
        self.val = float_spin(0.2, 0.0, 0.9, 0.05, 2)
        self.batch = int_spin(30, 1, 1024, "windows per training step")
        self.freeze = QCheckBox("freeze the CNN (fine-tune GRU + head only)")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        self.name = QLineEdit("finetune")
        self.run_btn, self.stop_btn = QPushButton("Rescale + train"), QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.runner.stop)
        self.status = QLabel("")
        self.plot = HistoryPlot()
        self.log = LogView()

        form = QFormLayout()
        form.addRow("start from", self.init)
        form.addRow("classes", self.classes)
        form.addRow("other data", self.extra)
        form.addRow("project movie x", self.oversample)
        form.addRow("epochs / lr", self._pair(self.epochs, self.lr))
        form.addRow("augmentation / non-event copies", self._pair(self.naug, self.add_nothing))
        form.addRow("validation share / batch", self._pair(self.val, self.batch))
        form.addRow("", self.freeze)
        form.addRow("device", self.device)
        form.addRow("run name", self.name)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.run_btn)
        lay.addWidget(self.stop_btn)
        lay.addWidget(self.status)
        lay.addWidget(self.plot)
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

    def set_init(self, path: str | Path) -> None:
        self.init.setText(str(path))

    def start(self) -> None:
        s = self.session
        try:
            s.require_movie()
            init = self.init.path()
            base = model_config(init).catnames if init else None
            catnames = build_catnames(base, self.classes.text())
            name = self.name.text().strip() or "finetune"
            run = s.run_dir(unused_name(s.project / "runs", name))
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Train", str(e))
            return
        data = s.prepared_dir
        cmds = [dextrusion_argv(*prepare_args(s.movie_path, data, s.project, s.cell_diameter,
                                              s.extrusion_duration))]
        extra = self.extra.path()
        if extra is not None:
            n = link_extra_data(extra, data)
            self.log.add(f"linked {n} file(s) of {extra} into {data}")
        dev = self.device.currentText()
        p = TrainParams(
            data, run, init, catnames, self.epochs.value(), self.lr.value(), self.naug.value(),
            self.add_nothing.value(), self.val.value(), self.batch.value(), self.freeze.isChecked(),
            {s.stem: self.oversample.value()} if self.oversample.value() > 1 else {},
            device=None if dev == "auto" else dev)
        cmds.append(dextrusion_argv(*train_args(p)))
        self._run = run
        run.mkdir(parents=True, exist_ok=True)
        self.plot.watch(run / "history.csv")
        self.status.setText(f"Training into {run}")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.runner.run(cmds)

    def _done(self, ok: bool) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.plot.stop_watching()
        if ok and self._run is not None:
            self.status.setText(f"Done: model saved in {self._run}")
            if self.on_trained:
                self.on_trained(self._run)
        else:
            self.status.setText("Training did not finish, see the log.")
