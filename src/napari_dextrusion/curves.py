"""Live training curves read from the ``history.csv`` written by ``dextrusion train``."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from qtpy.QtCore import QPointF, QRectF, Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QWidget


def read_history(path: str | Path) -> dict[str, list[float]]:
    """Columns of ``history.csv`` as float lists (empty if the file is missing / still empty)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: [float(r[k]) for r in rows if r.get(k) not in (None, "")]
            for k in (rows[0] if rows else {})}


def nice_range(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.05 or 0.05
    return lo - pad, hi + pad


class HistoryPlot(QWidget):
    """Loss (top) and accuracy (bottom) per epoch, training and validation, refreshed every 2 s."""

    PANELS = (("loss", ("loss", "val_loss"), "loss"), ("acc", ("acc", "val_acc"), "accuracy"))
    COLORS = {"loss": "#1f77b4", "val_loss": "#ff7f0e", "acc": "#1f77b4", "val_acc": "#ff7f0e"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 260)
        self.path: Path | None = None
        self.data: dict[str, list[float]] = {}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)

    def watch(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.data = {}
        if self.path:
            self._timer.start(2000)
        else:
            self._timer.stop()
        self.refresh()

    def stop_watching(self) -> None:
        self.refresh()
        self._timer.stop()

    def refresh(self) -> None:
        self.data = read_history(self.path) if self.path else {}
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = self.palette().window().color()
        fg = self.palette().windowText().color()
        p.fillRect(self.rect(), bg)
        h = self.height() / len(self.PANELS)
        for i, (_, keys, title) in enumerate(self.PANELS):
            self._panel(p, QRectF(40, i * h + 18, self.width() - 52, h - 38), keys, title, fg)
        p.end()

    def _panel(self, p: QPainter, r: QRectF, keys, title: str, fg: QColor) -> None:
        p.setPen(QPen(fg, 1))
        p.drawRect(r)
        p.drawText(QPointF(r.left(), r.top() - 5), title)
        epochs = self.data.get("epoch", [])
        allv = [v for k in keys for v in self.data.get(k, [])]
        if not epochs or not allv:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "no data yet")
            return
        lo, hi = nice_range(allv)
        x0, x1 = 1, max(len(epochs), 2)

        def pt(e, v):
            return QPointF(r.left() + (e - x0) / (x1 - x0) * r.width(),
                           r.bottom() - (v - lo) / (hi - lo) * r.height())

        p.drawText(QPointF(2, r.top() + 10), f"{hi:.3g}")
        p.drawText(QPointF(2, r.bottom()), f"{lo:.3g}")
        p.drawText(QPointF(r.right() - 40, r.bottom() + 13), f"ep {len(epochs)}")
        for j, key in enumerate(keys):
            vals = self.data.get(key, [])
            p.setPen(QPen(QColor(self.COLORS[key]), 2))
            prev = None
            for e, v in enumerate(vals, start=1):
                if not math.isfinite(v):
                    prev = None
                    continue
                q = pt(e, v)
                if prev is not None:
                    p.drawLine(prev, q)
                p.drawEllipse(q, 2, 2)
                prev = q
            p.drawText(QPointF(r.left() + 6 + j * 80, r.top() + 14), key)
