"""One review session: navigation over the queue, verdicts applied to the training folder.

GUI-free; the review tab is a thin layer on top of this class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .review import (
    EXCLUDE,
    NOTHING_SUFFIX,
    CropSpec,
    ReviewLog,
    apply_verdicts,
    extract_crop,
)


class ReviewRun:
    def __init__(self, log: ReviewLog, movie: np.ndarray, spec: CropSpec, folder: str | Path,
                 stem: str, on_applied=None):
        self.log, self.movie, self.spec = log, movie, spec
        self.folder, self.stem = Path(folder), stem
        self.on_applied = on_applied  # called after the training files changed
        pending = log.first_pending()
        self.index = pending if pending is not None else 0

    @property
    def n(self) -> int:
        return len(self.log)

    @property
    def entry(self) -> dict | None:
        return self.log.entries[self.index] if self.n else None

    @property
    def finished(self) -> bool:
        return self.log.first_pending() is None

    def crop(self) -> np.ndarray:
        e = self.entry
        return extract_crop(self.movie, self.spec, e["t"], e["y"], e["x"])

    def origin(self) -> tuple[int, int, int]:
        """Movie coordinates of the crop's first voxel (may be negative at the edges)."""
        e = self.entry
        return (e["t"] - self.spec.before, e["y"] - self.spec.half_y, e["x"] - self.spec.half_x)

    def verdict(self, value: str | None) -> None:
        """Record the verdict for the current event, write the files, move to the next pending."""
        if not self.n:
            return
        if value not in (None, EXCLUDE, NOTHING_SUFFIX) and value not in self.log.meta["classes"]:
            raise ValueError(f"unknown verdict {value!r}")
        self.log.set_verdict(self.index, value)
        apply_verdicts(self.log, self.folder, self.stem, self.movie.shape)
        if self.on_applied:
            self.on_applied()
        self._advance()

    def _advance(self) -> None:
        later = [i for i in range(self.index + 1, self.n) if self.log.entries[i]["verdict"] is None]
        earlier = [i for i in range(self.index) if self.log.entries[i]["verdict"] is None]
        if later or earlier:
            self.index = (later or earlier)[0]
        # all done: stay on the last event so that it can still be corrected with Back

    def back(self) -> None:
        self.index = max(0, self.index - 1)

    def skip(self) -> None:
        self.index = min(self.n - 1, self.index + 1)

    def progress(self) -> str:
        done = sum(e["verdict"] is not None for e in self.log.entries)
        return f"{self.index + 1}/{self.n}   reviewed {done}"
