import geobench
from torch.utils.data import Dataset, DataLoader
import glob
import h5py
import json
import os
import torch
import numpy as np
import json 
from MODALITIES import *
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn as nn



##################### FUNCTIONS FOR PRETRAINING DATASETS #####################

def build_pretraining_dataset(is_train, args):
    # transform = build_transform(is_train, args)
    split = "train" if is_train else "val"
    dataset = multimodal_dataset(args, split=split)
    return dataset

class multimodal_dataset(Dataset):
    def __init__(self, args, split = "train"):
        # self.data_name = args.data_name #name of the dataset. for example: data_100k_130
        self.data_path = args.data_path # path to the dataset
        self.data_name = args.data_name
        self.splits_path = args.splits_path # path to the split file
        self.tile_info = args.tile_info # tile info
        self.modalities = args.modalities # modalities used for training
        self.modalities_full = args.modalities_full # all modalities present in the datasets. This is used to keep track of the indices of the modalities in the dataset.
        self.indices = json.load(open(self.splits_path, 'r'))[split]
        self.random_crop = args.random_crop
        self.random_crop_size = args.input_size


        self.norm_stats = args.band_stats # mean, std, min and max of each band

    def transform_random_crop(self, return_dict, random_crop_size=112):
        # applying random crop for every modality
        for modality in return_dict:
            # we only random crop for pixel based modalities
            if modality in ['sentinel2', 'sentinel1', 'aster', 'canopy_height_eth', 'dynamic_world', 'esa_worldcover']:
                c, h, w = return_dict[modality].shape
                i, j, h, w = transforms.RandomCrop.get_params(return_dict[modality], output_size=(random_crop_size, random_crop_size))
                return_dict[modality] = TF.crop(return_dict[modality], i, j, h, w)
            else:
                return_dict[modality] = return_dict[modality]
        return return_dict

    def _open_hdf5(self, path):
        self.data_full = h5py.File(path, 'r')

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        # this is to ensure that multiple workers do not open the same file multiple times.
        if not hasattr(self, 'data_full'):
            self._open_hdf5(self.data_path)

        # based on what bands and what modalities we need for training, we return the data[idx].)
        return_dict = {}
        name = self.data_full['metadata'][self.indices[idx]][0].decode('utf-8')
        l2a = self.tile_info[name]['S2_type'] == 'l2a'

        for modality in self.modalities.keys():
            # get the indices based on how it is in modalities_full
            if self.modalities[modality] == 'all':
                modality_idx = [i for i in range(len(self.modalities_full[modality]))]
            else:
                modality_idx = [self.modalities_full[modality].index(m) for m in self.modalities[modality]]
            
            if modality in ['biome', 'eco_region']:
                # for these modalities the array is already one hot encoded. hence modality_idx is not needed.
                data = self.data_full[modality][self.indices[idx], ...]
                data = np.array(data)
            else:
                # get the data
                data = self.data_full[modality][self.indices[idx], modality_idx, ...]
                data = np.array(data)



            # inside the band_stats, the name for sentinel2 is sentinel2_l1c or sentinel2_l2a
            if modality == 'sentinel2':
                modality_ = 'sentinel2_l2a' if l2a else 'sentinel2_l1c'
            else:
                modality_ = modality

            if modality not in ['biome', 'eco_region', 'dynamic_world', 'esa_worldcover']:
                means = np.array(self.norm_stats[modality_]['mean'])[modality_idx]
                stds = np.array(self.norm_stats[modality_]['std'])[modality_idx]
                if modality in ['era5', 'lat', 'lon', 'month']:
                    # single value mean and std
                    data = (data - means)/stds
                else:
                    # single value mean and std for each band 
                    data = (data - means[:, None, None]) / stds[:, None, None]
                    
            if modality == 'dynamic_world':
                # the labels of dynamic world are 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, nan respectively.
                # originally when downloading the no data values are 0. hence we remap them to nan.
                data = np.where(data == NO_DATA_VAL[modality], np.nan, data)
                old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 8, we map them to nan
                data = np.where(data > 8, np.nan, data)

            if modality == 'esa_worldcover':
                # the labels of esa worldcover are 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255. We convert them to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 255 respectively.
                old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
                new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]
                for old, new in zip(old_values, new_values):
                    data = np.where(data == old, new, data)
                # for any value greater than 10, we map them to nan
                data = np.where(data > 10, np.nan, data)
                
            
            # converting the nodata values to nan to keep everything consistent
            data = np.where(data == NO_DATA_VAL[modality], np.nan, data) if modality != 'dynamic_world' else data
            data = torch.from_numpy(data).float()
            return_dict[modality] = data

        if self.random_crop:
            return_dict = self.transform_random_crop(return_dict, random_crop_size=self.random_crop_size)

        # we also return the id, to differentiate between sentinel2_l1c and sentinel2_l2a, since this is given in the tile_info json file. To keep everything 
        # consistent, we name the modality as sentinel2 instead of sentinel2_l1c or sentinel2_l2a
        return_dict['id'] = name
        return return_dict
    


##################### FUNCTIONS FOR FINE-TUNING DATASETS #####################

BAND_NAMES = json.load(open("BAND_NAMES.json", "r"))

class geobench_dataset(Dataset):
    def __init__(self, dataset_name = None, split = "train", transform = None, benchmark_name = "classification"):
        if split == "val":
            split = "valid"

        if benchmark_name == "classification":
            benchmark_name = "classification_v0.9.1/"
        elif benchmark_name == "segmentation":
            benchmark_name = "segmentation_v0.9.1/"

        for task in geobench.task_iterator(benchmark_name=benchmark_name):
            if task.dataset_name == dataset_name:
                break
        self.transform = transform
        self.dataset_name = dataset_name
        self.dataset = task.get_dataset(split=split, band_names=BAND_NAMES[dataset_name])
        self.label_map = task.get_label_map()
        self.label_stats = task.label_stats() if benchmark_name != "segmentation_v0.9.1/" else "None"
        self.dataset_dir = task.get_dataset_dir()
        if dataset_name == "m-brick-kiln":
            self.num_classes = 2
        elif dataset_name == "m-bigearthnet":
            self.num_classes = 43
        elif dataset_name == "m-cashew-plantation":
            self.num_classes = 7
        elif dataset_name == "m-SA-crop-type":
            self.num_classes = 10
        else:
            self.num_classes = len(task.get_label_map().keys())
        self.tmp_band_names = [self.dataset[0].bands[i].band_info.name for i in range(len(self.dataset[0].bands))]
        # get the tmp bands in the same order as the ones present in the BAND_NAMES.json file
        self.tmp_band_indices = [self.tmp_band_names.index(band_name) for band_name in BAND_NAMES[dataset_name]]
        self.norm_stats = self.dataset.normalization_stats()
        self.in_channels = len(self.tmp_band_indices)


        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        label = self.dataset[idx].label
        x = []

        for band_idx in self.tmp_band_indices:
            x.append(self.dataset[idx].bands[band_idx].data)

        x = np.stack(x, axis=0)

        mean = np.array(self.norm_stats[0])
        std = np.array(self.norm_stats[1])

        if self.dataset_name == "m-so2sat":
            # the mean and std are multiplied by 10000 only for the so2sat dataset, while the 
            # data values are in decimal range between 0 and 1. Hence, we need to divide the mean and std by 10000
            mean = mean/10000
            std = std/10000

        # normalize each band with its mean and std
        x = (x - mean[:, None, None]) / std[:, None, None]
        x = torch.from_numpy(x).float()


        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        return x, label, mean, std
    

class geobench_dataset_subset(Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.in_channels = dataset.in_channels
        self.num_classes = dataset.num_classes
        self.norm_stats = dataset.norm_stats
        self.dataset_name = dataset.dataset_name

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/qbk152/vishal/global-lr/data/data_100k_130/data_100k_130.h5")
    parser.add_argument("--splits_path", type=str, default="/home/qbk152/vishal/global-lr/data/data_100k_130/data_100k_130_splits.json")
    parser.add_argument("--tile_info_path", type=str, default="/home/qbk152/vishal/global-lr/data/data_100k_130/data_100k_130_tile_info.json")
    parser.add_argument("--band_stats", type=str, default="/home/qbk152/vishal/global-lr/data/data_100k_130/data_100k_130_band_stats.json")
    parser.add_argument("--data_name", type=str, default="data_100k_130")
    args = parser.parse_args()


    args.modalities = MODALITIES
    args.modalities_full = MODALITIES_FULL
    args.band_stats = json.load(open(args.band_stats, 'r'))
    dataset = multimodal_dataset(args, split="train")

    print(len(dataset))
    data = dataset[0]
    print(data.keys())
    print(data['sentinel2_l1c'])
