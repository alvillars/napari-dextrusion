"""Small Qt helpers shared by the tabs."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)


def int_spin(value: int, lo: int, hi: int, tip: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    w.setToolTip(tip)
    return w


def float_spin(value: float, lo: float, hi: float, step: float = 1.0, decimals: int = 2,
               tip: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    w.setToolTip(tip)
    return w


class PathRow(QWidget):
    """A line edit with a Browse button (folder or file)."""

    def __init__(self, folder: bool = True, caption: str = "Choose", filt: str = "", parent=None):
        super().__init__(parent)
        self.folder, self.caption, self.filt = folder, caption, filt
        self.edit = QLineEdit()
        button = QPushButton("Browse…")
        button.clicked.connect(self._browse)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(button)

    def _browse(self) -> None:
        start = self.edit.text() or str(Path.home())
        if self.folder:
            path = QFileDialog.getExistingDirectory(self, self.caption, start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, self.caption, start, self.filt)
        if path:
            self.edit.setText(path)

    def text(self) -> str:
        return self.edit.text().strip()

    def path(self) -> Path | None:
        return Path(self.text()) if self.text() else None

    def setText(self, text: str) -> None:
        self.edit.setText(str(text))


class LogView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setMinimumHeight(110)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

    def add(self, text: str) -> None:
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
