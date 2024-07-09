import shutil
from pathlib import Path

import pytest

from main_pretrain import main, get_args_parser


@pytest.fixture
def args():
    args = get_args_parser()
    args = args.parse_args(args=[])
    args.epochs = 1
    args.batch_size = 2
    args.debug = True
    args.device = "cpu"
    args.sparse = False

    return args

@pytest.mark.parametrize(
    "mod_setting",
    [
        "full",
        "s2_only",
        "s2_rgb"
    ],
)
def test_mmearth_pretrain(args, mod_setting):
    test_out = Path("test_out")
    test_out.mkdir(exist_ok=False)
    args.mod_setting = mod_setting
    args.processed_dir = test_out

    try:
        main(args)
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)
