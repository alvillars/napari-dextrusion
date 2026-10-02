"""The plugin's dock widget: open a movie, then Detect / Annotate / Train / Review tabs."""

from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .session import Session
from .tab_annotate import AnnotateTab
from .tab_inference import InferenceTab
from .tab_review import ReviewTab
from .tab_train import TrainTab
from .ui import float_spin


class DextrusionWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.session = Session()

        open_btn = QPushButton("Open movie (.tif, T×Y×X)…")
        open_btn.clicked.connect(self.open_dialog)
        layer_btn = QPushButton("Use the selected image layer")
        layer_btn.setToolTip("for a movie opened with napari: it must come from a .tif file")
        layer_btn.clicked.connect(self.use_selected_layer)
        self.movie_label = QLabel("no movie")
        self.movie_label.setWordWrap(True)
        self.diameter = float_spin(25, 3, 500, 5, 1, "typical cell diameter in this movie (pixels)")
        self.duration = float_spin(4.5, 1, 100, 0.5, 1,
                                   "typical duration of an event in this movie (frames)")
        self.diameter.valueChanged.connect(lambda v: setattr(self.session, "cell_diameter", v))
        self.duration.valueChanged.connect(lambda v: setattr(self.session, "extrusion_duration", v))
        scale = QFormLayout()
        scale.addRow("cell diameter (px)", self.diameter)
        scale.addRow("event duration (frames)", self.duration)

        self.tabs = QTabWidget()
        self.detect = InferenceTab(self.viewer, self.session)
        self.annotate = AnnotateTab(self.viewer, self.session)
        self.train = TrainTab(self.viewer, self.session, on_trained=self._trained)
        self.review = ReviewTab(self.viewer, self.session,
                                goto_train=lambda: self.tabs.setCurrentWidget(self.train))
        for tab, name in ((self.detect, "Detect"), (self.annotate, "Annotate"),
                          (self.train, "Train"), (self.review, "Review")):
            self.tabs.addTab(tab, name)

        lay = QVBoxLayout(self)
        lay.addWidget(open_btn)
        lay.addWidget(layer_btn)
        lay.addWidget(self.movie_label)
        lay.addLayout(scale)
        lay.addWidget(self.tabs, 1)

    # ---------------------------------------------------------------------------- movie
    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a 2D+t movie", "", "TIFF (*.tif *.tiff)")
        if path:
            self.open_movie(path)

    def use_selected_layer(self) -> None:
        layer = self.viewer.layers.selection.active
        src = getattr(getattr(layer, "source", None), "path", None)
        if layer is None or not src or layer.ndim != 3:
            QMessageBox.warning(self, "Open movie", "select a (T, Y, X) image layer read from a .tif")
            return
        self.open_movie(src)

    def open_movie(self, path) -> None:
        try:
            img = self.session.open_movie(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Open movie", str(e))
            return
        if "movie" in self.viewer.layers:
            self.viewer.layers.remove("movie")
        lo, hi = np.percentile(img[:: max(1, len(img) // 10)], (1, 99.8))
        self.viewer.add_image(img, name="movie", contrast_limits=(float(lo), float(hi)))
        self.movie_label.setText(f"{self.session.movie_path.name}  {img.shape}\n"
                                 f"project folder: {self.session.project}")

    # ------------------------------------------------------------------------- training
    def _trained(self, run) -> None:
        """A model was trained: offer it to the detection tab."""
        self.detect.set_model(run)
        self.train.set_init(run)
