import json
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Union


import h5py
import numpy as np

try:
    import ffcv
    from ffcv import DatasetWriter
    from ffcv.fields import NDArrayField
    from ffcv.fields.ndarray import NDArrayDecoder
    from ffcv.loader import OrderOption
    from ffcv.transforms import ToTensor
except ImportError:
    print("FFCV not installed, please install it to use the beton file creation.")

import torch
from torch.utils.data import Dataset

import MODALITIES


##################### FUNCTIONS FOR PRETRAINING DATASETS #####################


def build_pretraining_dataset(is_train, args):
    split = "train" if is_train else "val"
    dataset = MMEarthDataset(args, split=split)
    return dataset


class MMEarthDataset(Dataset):
    def __init__(self, args, split="train", return_tuple: bool = False):
        # self.data_name = args.data_name #name of the dataset. for example: data_100k_130
        self.data_path = args.data_path  # path to the dataset
        self.data_name = args.data_name
        self.splits_path = args.splits_path  # path to the split file
        self.tile_info = args.tile_info  # tile info
        self.modalities = args.modalities  # modalities used for training
        self.modalities_full = (
            args.modalities_full
        )  # all modalities present in the datasets. This is used to keep track of the indices of the modalities in the dataset.
        self.indices = json.load(open(self.splits_path, "r"))[split]
        self.return_tuple = return_tuple

        self.norm_stats = args.band_stats  # mean, std, min and max of each band

    def _open_hdf5(self, path):
        self.data_full = h5py.File(path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # this is to ensure that multiple workers do not open the same file multiple times.
        if not hasattr(self, "data_full"):
            self._open_hdf5(self.data_path)

        # based on what bands and what modalities we need for training, we return the data[idx].)
        return_dict = OrderedDict()
        name = self.data_full["metadata"][self.indices[idx]][0].decode("utf-8")
        l2a = self.tile_info[name]["S2_type"] == "l2a"

        for modality in self.modalities.keys():
            # get the indices based on how it is in modalities_full
            if self.modalities[modality] == "all":
                modality_idx = [i for i in range(len(self.modalities_full[modality]))]
            else:
                modality_idx = [
                    self.modalities_full[modality].index(m)
                    for m in self.modalities[modality]
                ]

            if modality in ["biome", "eco_region"]:
                # for these modalities the array is already one hot encoded. hence modality_idx is not needed.
                data = self.data_full[modality][self.indices[idx], ...]
                data = np.array(data)
            else:
                # get the data
                data = self.data_full[modality][self.indices[idx], modality_idx, ...]
                data = np.array(data)

            # inside the band_stats, the name for sentinel2 is sentinel2_l1c or sentinel2_l2a
            if modality == "sentinel2":
                modality_ = "sentinel2_l2a" if l2a else "sentinel2_l1c"
            else:
                modality_ = modality

            if modality not in [
                "biome",
                "eco_region",
                "dynamic_world",
                "esa_worldcover",
            ]:
                means = np.array(self.norm_stats[modality_]["mean"])[modality_idx]
                stds = np.array(self.norm_stats[modality_]["std"])[modality_idx]
                if modality in ["era5", "lat", "lon", "month"]:
                    # single value mean and std
                    data = (data - means) / stds
                else:
                    # single value mean and std for each band
                    data = (data - means[:, None, None]) / stds[:, None, None]

            if modality == "dynamic_world":
                # the labels of dynamic world are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, nan respectively.
                # originally when downloading the no data values are 0. hence we remap them to nan.
                data = np.where(data == MODALITIES.NO_DATA_VAL[modality], np.nan, data)
                old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 8, we map them to nan
                data = np.where(data > 8, np.nan, data)

            if modality == "esa_worldcover":
                # the labels of esa worldcover are 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 255 respectively.
                old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 10, we map them to nan
                data = np.where(data > 10, np.nan, data)

            # converting the nodata values to nan to keep everything consistent
            data = (
                np.where(data == MODALITIES.NO_DATA_VAL[modality], np.nan, data)
                if modality != "dynamic_world"
                else data
            )

            if MODALITIES.MODALITY_TASK[modality] in ["classification", "segmentation"]:
                # remap nan values to -1
                data = np.where(np.isnan(data), -1, data)
                data = data.astype(np.dtype("int64"))
            else:
                data = data.astype(np.dtype("float32"))

            return_dict[modality] = data

        # we also return the id, to differentiate between sentinel2_l1c and sentinel2_l2a, since this is given in the tile_info json file.
        return_dict["id"] = name
        # To keep everything consistent, we name the modalities as sentinel2 instead of sentinel2_l1c or sentinel2_l2a

        if self.return_tuple:
            return tuple(return_dict.values())

        return return_dict


def get_single_glob_file(data_root: Path, pattern) -> Path:
    file = [f for f in data_root.glob(pattern)]
    assert len(file) < 2, f"too many {pattern} files at {data_root}"
    assert len(file) > 0, f"no {pattern} files at {data_root}"
    return file[0]


def create_MMEearth_args(data_root: Path, modalities: dict) -> Namespace:
    args = Namespace()

    args.data_path = get_single_glob_file(data_root, "data_*.h5")
    args.splits_path = get_single_glob_file(data_root, "data_*_splits.json")
    args.tile_info_path = get_single_glob_file(data_root, "data_*_tile_info.json")
    with open(args.tile_info_path, "r") as f:
        args.tile_info = json.load(f)
    args.band_stats_path = get_single_glob_file(data_root, "data_*_band_stats.json")
    with open(args.band_stats_path, "r") as f:
        args.band_stats = json.load(f)
    args.data_name = data_root.name
    args.modalities = modalities
    args.modalities_full = MODALITIES.MODALITIES_FULL
    return args


def get_mmearth_dataloaders(
    data_dir: Path,
    processed_dir: Path,
    modalities: dict,
    num_workers: int,
    batch_size_per_device: int,
    splits: list[str] = None,
    no_ffcv: bool = False,
    indices: list[list[int]] = None,
    distributed: bool = False,
) -> list[Union[ffcv.Loader]]:
    """
    Creates and returns data loaders for the MMEarth dataset. If the processed beton file does not exist, it processes the data
    and creates the beton file, then returns FFCV data loaders.

    Parameters:
    ----------
    data_dir : Path
        The directory where the raw dataset is stored.
    processed_dir : Path
        The directory where the processed beton files will be saved.
    modalities : dict
        A dictionary specifying the modalities configurations.
    num_workers : int
        The number of worker threads to use for data loading.
    batch_size_per_device : int
        The batch size for each device during training.
    splits : list[str], optional
        The dataset splits to be used. Default is ["train", "val"].
    indices: list[list[int]], optional
        Select indices to use for each split (starting at 0). Default is None, meaning all samples are used. Only with FFCV enabled.
    distributed: bool, optional
        Decides if RandomSampler (False) or QuasiRandomSampler (True) is used.

    Returns:
    -------
    list[Union[ffcv.Loader, torch.utils.data.DataLoader]]
        A list containing data loaders. Each loader can be either `ffcv.Loader` (for beton files) or `torch.data.DataLoader` (for standard PyTorch datasets).


    Example Usage:
    --------------
    ```python
    from pathlib import Path

    data_dir = Path("/path/to/raw/data")
    processed_dir = Path("/path/to/processed/data")
    input_modality = {...}  # Define your input modalities configurations
    target_modality = {...}  # Define your target modalities configurations
    num_workers = 4
    batch_size_per_device = 32

    dataloaders = get_mmearth_dataloaders(
        data_dir,
        processed_dir,
        modalities,
        num_workers,
        batch_size_per_device,
        splits=["train"]
    )
    ```

    Notes:
    -----
    - The function checks if the processed beton file exists for each split. If it doesn't exist, it processes the data
      and creates the beton file.
    - The input and target modalities are reverse looked up using `IN_MODALITIES` and `MODALITIES_FULL` respectively.
    - The `convert_mmearth` function is used to convert the dataset into beton format.
    - The `ffcv.Loader` is used to create the data loaders with appropriate pipelines for training and validation.

    """
    if splits is None:
        splits = ["train"]
    assert not no_ffcv or (
        no_ffcv and indices is None
    ), "Providing indices is not supported in no_ffcv mode."
    assert indices is None or (len(indices) == len(splits)), (
        "If indices are given, the number of splits and number of list of indices"
        "must align (len(indices) != len(splits) = ({len(indices)} != {len(splits))}"
    )

    if processed_dir is None:
        processed_dir = data_dir
    else:
        processed_dir.mkdir(exist_ok=True)

    dataloaders = []
    for i, split in enumerate(splits):
        is_train = split == "train"
        subset = "" if indices is None else "_subset"
        beton_file = processed_dir / f"{split}{subset}.beton"
        args = create_MMEearth_args(data_dir, modalities)

        if no_ffcv:
            
            dataset = MMEarthDataset(args, split=split, return_tuple=False)
            
            dataloaders.append(dataset)
            continue
        if not beton_file.exists():
            print(
                f"Processed file {beton_file} does not exist, trying to create it now."
            )
            dataset = MMEarthDataset(args, split=split, return_tuple=True)

            if len(dataset) == 0:
                assert not is_train, "training dataset has no samples"
                print(f"No samples in evaluation split '{split}', skipping it")
                dataloaders.append(None)
                continue

            idx = None if indices is None else indices[i]
            convert_mmearth_to_beton(
                dataset,
                beton_file,
                num_workers=num_workers,
                modalities=args.modalities,
                indices=idx,
            )

        # Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
        sampler = (
            OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        )  # quasi random not working distributed
        sampler = sampler if is_train else OrderOption.SEQUENTIAL
        dataloader = ffcv.Loader(
            beton_file,
            batch_size=batch_size_per_device,
            num_workers=num_workers,
            order=sampler,
            drop_last=is_train,
            distributed=distributed,
        )

        dataloaders.append(dataloader)

    return dataloaders


def convert_mmearth_to_beton(
    dataset: MMEarthDataset,
    write_path: Path,
    modalities: OrderedDict,
    num_workers: int = -1,
    indices: list = None,
):
    """
    Converts a MMEarth dataset into a format optimized for a specified machine learning task and writes it to a specified path.

    Parameters:
    ----------
    dataset : MMEarthDataset
        The dataset to be converted and written. It should be compatible with the DatasetWriter's from_indexed_dataset method.
    write_path : Path
        The file path where the transformed dataset will be written.
    modalities : OrderedDict
        All modalities that are returned (in order).
    num_workers : int, optional
        The number of worker threads to use for writing the dataset. A value of -1 indicates that the default number of workers should be used. Default is -1.
    indices : list, optional
        Indices to select from dataset, good for subset creation.


    Process:
    -------
    1. Field Initialization:
        Initializes the fields dictionary with a sentinel2 field.
        Adds a label field to the fields dictionary based on the supervised_task.
    2. Dataset Writing:
        Creates a DatasetWriter instance with the specified write_path, fields, and num_workers.
        Writes the dataset using the from_indexed_dataset method of the DatasetWriter.

    """

    # get one example batch to infer shapes
    sample = dataset[0]

    fields = OrderedDict()
    for i, name in enumerate(modalities):
        x = sample[i]
        dtype = x.dtype
        assert dtype in [np.dtype("float32"), np.dtype("int64")]
        shape = x.shape

        fields[name] = NDArrayField(dtype=dtype, shape=shape)
    # fields["id"] = NDArrayField(dtype=np.dtype("int64"), shape=(1,))

    # Pass a type for each data field
    writer = DatasetWriter(write_path, fields, num_workers=num_workers)

    # Write dataset
    writer.from_indexed_dataset(dataset, indices=indices)
