import shutil
from pathlib import Path

import numpy as np
import pytest

import MODALITIES

from geobenchdataset import GeobenchDataset
from mmearth_dataset import MMEarthDataset, create_MMEearth_args, get_mmearth_dataloaders


@pytest.mark.parametrize(
    "modalities",
    [MODALITIES.INP_MODALITIES, MODALITIES.RGB_MODALITIES],
)
def test_mmearth_dataset(modalities):
    split = "train"
    args = create_MMEearth_args(MODALITIES.MMEARTH_DIR, modalities)

    dataset = MMEarthDataset(args, split=split)

    if split == "train":
        assert len(dataset) > 0, "Dataset should not be empty"
        data = dataset[0]
        assert "sentinel2" in data, "Dataset should contain 'sentinel2' key"
        s1_channel = 8
        s2_channel = 12
        if modalities == MODALITIES.OUT_MODALITIES:
            assert isinstance(
                data["sentinel1"], np.ndarray
            ), "'sentinel1' data should be a Tensor"
            assert (
                    data["sentinel1"].shape[0] == s1_channel
            ), f"'sentinel1' data should have {s1_channel} channels"
        elif modalities == MODALITIES.RGB_MODALITIES:
            s2_channel = 3
        assert isinstance(
            data["sentinel2"], np.ndarray
        ), "'sentinel2' data should be a Tensor"
        assert (
                data["sentinel2"].shape[0] == s2_channel
        ), f"'sentinel2' data should have {s2_channel} channels"

    # no tests for val/test currently


@pytest.mark.parametrize(
    "modalities",
    [MODALITIES.INP_MODALITIES, MODALITIES.RGB_MODALITIES],
)
def test_mmearth_dataloader(modalities):
    test_out = Path("test_out")
    test_out.mkdir(exist_ok=True)

    try:
        loader = get_mmearth_dataloaders(
            MODALITIES.MMEARTH_DIR,
            test_out,
            modalities,
            2,
            2,
            ["train"],
            indices=[list(range(10))],
        )

        for data in loader:
            break
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)


@pytest.mark.parametrize("split", ["train", "val", "test"])
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
def test_geobench_dataset(split, dataset_name):
    dataset = GeobenchDataset(
        dataset_name=dataset_name,
        split=split,
        transform=None,
    )

    assert len(dataset) > 0, f"Dataset '{dataset_name}' should not be empty"

    n_channel = dataset[0][0].shape[0]
    expected = 12
    if dataset_name == "m-brick-kiln":
        expected = 3
    assert (
            n_channel == expected
    ), f"Dataset '{dataset_name}' should have {expected} channels, found {n_channel}"


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
@pytest.mark.parametrize(
    "no_ffcv",
    [False, True],
)
def test_geobench_dataloader(dataset_name, no_ffcv):
    test_out = Path("test_out")
    test_out.mkdir(exist_ok=True)
    splits = ["train", "val", "test"]
    partition = "default"

    try:
        loaders, task = get_geobench_dataloaders(
            dataset_name,
            test_out,
            2,
            2,
            splits,
            partition,
            no_ffcv,
            indices=None if no_ffcv else [list(range(10)), list(range(10)), list(range(10))],
        )
        for loader in loaders:
            for data in loader:
                break
    finally:
        # cleanup
        shutil.rmtree(test_out, ignore_errors=True)
