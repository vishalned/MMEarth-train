import shutil
from pathlib import Path

from MODALITIES import MMEARTH_DIR, OUT_MODALITIES
from mmearth_dataset import MMEarthDataset, create_MMEearth_args, convert_mmearth_to_beton


def test_mmearth_dataset():
    split = "train"
    modalities = OUT_MODALITIES

    args = create_MMEearth_args(MMEARTH_DIR, modalities)

    dataset = MMEarthDataset(args, split=split, return_tuple=True)

    test_out = Path("test_out")
    test_out.mkdir(exist_ok=False)
    write_path = test_out / "mmearth.beton"

    try:
        convert_mmearth_to_beton(dataset, write_path, modalities, num_workers=1, indices=[i for i in range(10)])
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)
