import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

EVENTS = [(10, 40, 45), (14, 70, 30), (20, 60, 70), (24, 30, 65)]


@pytest.fixture(scope="session")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def make_movie(t=32, size=96, seed=1):
    """uint16 (T, Y, X): noise plus a bright spot around every event of EVENTS."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    movie = rng.random((t, size, size)).astype(np.float32) * 0.3
    for t0, y0, x0 in EVENTS:
        for i in range(t):
            rad = 2 + 0.7 * max(0, 5 - abs(i - t0))
            movie[i] += np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * rad**2))
    return (movie / movie.max() * 4000).astype(np.uint16)


@pytest.fixture()
def movie_file(tmp_path):
    import tifffile

    path = tmp_path / "data" / "mov.tif"
    path.parent.mkdir()
    tifffile.imwrite(path, make_movie())
    return path


@pytest.fixture()
def tiny_model(tmp_path):
    from dextrusion.config import DeXConfig
    from dextrusion.io import save_model
    from dextrusion.model import DeXNet

    cfg = DeXConfig(nb_filters=4)
    path = tmp_path / "net"
    save_model(path, DeXNet(cfg.ncat, 4), cfg)
    return path


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    """Modal dialogs would block forever without a user: record the warnings instead."""
    from qtpy.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]) or 0)
    return shown
