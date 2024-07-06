import shutil
from pathlib import Path

import pytest

from main_finetune import main, get_args_parser


@pytest.fixture
def args():
    args = get_args_parser()
    args = args.parse_args(args=[])
    args.epochs = 1
    args.batch_size = 2
    args.debug = True
    args.device = "cpu"
    args.version = "1.0"

    return args

@pytest.mark.parametrize(
    "dataset_name",
    [
        "m-eurosat",
        "m-so2sat",
        "m-bigearthnet",
        "m-brick-kiln",
        "m-cashew-plant",
        "m-SA-crop-type",
    ],
)
def test_mmearth_dataloader(args, dataset_name):
    test_out = Path("test_out")
    test_out.mkdir(exist_ok=False)
    args.processed_dir = test_out

    try:
        main(args)
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)
