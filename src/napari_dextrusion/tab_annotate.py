"""Annotation tab: one Points layer per class, saved as ROI files in the project folder.

This is the annotation tool of ``dextrusion label`` (``dextrusion.label``), embedded in the plugin.
"""

from __future__ import annotations

from dextrusion.label import (
    CLASS_KEYS,
    DEFAULT_CLASSES,
    attach_controls,
    build_layers,
    load_points,
    roi_path,
)
from qtpy.QtWidgets import QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .review import class_label, class_suffix
from .session import Session


def parse_class_names(text: str) -> dict[str, str]:
    """``'division, delamination'`` -> ``{'division': '_cell_division.zip', ...}`` (ordered)."""
    names = [n.strip() for n in text.split(",") if n.strip()]
    if not names:
        raise ValueError("give at least one class name")
    if len(names) > len(CLASS_KEYS):
        raise ValueError(f"at most {len(CLASS_KEYS)} classes can be annotated at once")
    if len(set(names)) != len(names):
        raise ValueError("duplicate class names")
    return {n: class_suffix(n) for n in names}


class AnnotateTab(QWidget):
    def __init__(self, viewer, session: Session):
        super().__init__()
        self.viewer, self.session = viewer, session
        self.layers: dict = {}
        self.save_all = None
        self.controls = None
        self.classes = QLineEdit(", ".join(DEFAULT_CLASSES))
        self.classes.setToolTip("class names, comma separated; each is saved as <movie>_cell_<name>.zip")
        self.start_btn = QPushButton("Start annotating")
        self.start_btn.clicked.connect(self.start)
        hint = QLabel("Annotations are ROI files in the project folder (movie coordinates). "
                      "Mark each event at the frame and cell centre where it is most recognisable.")
        hint.setWordWrap(True)
        self.holder = QVBoxLayout()
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(QLabel("classes"))
        lay.addWidget(self.classes)
        lay.addWidget(self.start_btn)
        lay.addLayout(self.holder)
        lay.addStretch(1)

    def set_classes_from_suffixes(self, suffixes: list[str]) -> None:
        self.classes.setText(", ".join(class_label(s) for s in suffixes if s))

    def start(self) -> None:
        try:
            self.session.require_movie()
            classes = parse_class_names(self.classes.text())
        except Exception as e:  # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Annotate", str(e))
            return
        if self.layers:  # restart with other classes: save and drop the previous layers
            self.save_all()
            for layer in self.layers.values():
                self.viewer.layers.remove(layer)
            self.controls.setParent(None)
        if "movie" in self.viewer.layers:
            self.viewer.layers.remove("movie")
        s = self.session
        self.layers, self.save_all = build_layers(
            self.viewer, s.movie, s.project, s.stem, classes, point_size=s.cell_diameter)
        self.controls = attach_controls(self.viewer, self.layers, self.save_all, list(classes))
        self.holder.addWidget(self.controls)
        self.classes_dict = classes
        s.reload_annotations = self.reload

    def reload(self) -> None:
        """Re-read the ROI files (after the review tab changed them) into the layers."""
        if not self.layers:
            return
        for name, layer in self.layers.items():
            suffix = self.classes_dict[name]
            layer.data = load_points(roi_path(self.session.project, self.session.stem, suffix))
