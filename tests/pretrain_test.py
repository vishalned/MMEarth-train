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


def test_mmearth_dataloader(args):
    test_out = Path("test_out")
    test_out.mkdir(exist_ok=False)

    try:
        main(args)
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)
