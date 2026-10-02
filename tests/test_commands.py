from pathlib import Path

from dextrusion.cli import build_parser

from napari_dextrusion.commands import (
    DetectParams,
    TrainParams,
    detect_args,
    dextrusion_argv,
    prepare_args,
    train_args,
)


def parse(args):
    return build_parser().parse_args(args)


def test_detect_args_are_accepted_by_the_cli_parser():
    p = DetectParams(Path("models/notum_all"), Path("out"), cell_diameter=50, device="cpu")
    a = parse(detect_args("m.tif", p))
    assert a.cell_diameter == 50 and a.device == "cpu" and a.dxy == 10 and a.models == p.models
    assert parse(detect_args("m.tif", DetectParams(Path("m"), Path("o")))).device is None


def test_prepare_args_are_accepted_by_the_cli_parser():
    a = parse(prepare_args("m.tif", "prep", "ann", 50, 4.5))
    assert a.cell_diameter == 50 and a.rois_dir == Path("ann") and a.out == Path("prep")


def test_train_args_are_accepted_by_the_cli_parser():
    p = TrainParams(Path("prep"), Path("run"), init_from=Path("models/notum_all/notumAll0"),
                    catnames=["", "_cell_death.zip", "_cell_division.zip"], epochs=7,
                    freeze_cnn=True, oversample={"m": 30}, device="cpu")
    a = parse(train_args(p))
    assert a.epochs == 7 and a.freeze_cnn and a.oversample == ["m=30"]
    assert a.catnames == ["", "_cell_death.zip", "_cell_division.zip"]  # empty first name kept
    assert a.add_nothing == 10  # hard negatives are used by default
    plain = parse(train_args(TrainParams(Path("d"), Path("o"))))
    assert plain.init_from is None and plain.catnames is None and not plain.freeze_cnn


def test_dextrusion_argv_runs_the_real_entry_point():
    import subprocess

    out = subprocess.run(dextrusion_argv("detect", "--help"), capture_output=True, text=True)
    assert out.returncode == 0 and "--cell-diameter" in out.stdout
