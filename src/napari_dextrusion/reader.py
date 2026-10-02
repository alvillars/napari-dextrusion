"""npe2 reader: open a (T, Y, X) .tif movie as an image layer."""

from __future__ import annotations

from pathlib import Path


def napari_get_reader(path):
    """Return a reader for a single 3D .tif file, else ``None`` (napari then tries other readers)."""
    if isinstance(path, list) or not str(path).lower().endswith((".tif", ".tiff")):
        return None
    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            if len(tif.series[0].shape) != 3:
                return None
    except Exception:  # noqa: BLE001 - not a readable tif: let another reader report it
        return None
    return read_movie_layer


def read_movie_layer(path):
    from dextrusion.io import read_movie

    img = read_movie(path)
    return [(img, {"name": Path(path).stem}, "image")]
