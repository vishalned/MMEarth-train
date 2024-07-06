# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import numpy as np

def build_dataset(split, args, percent = None, num_samples = None):
    is_train = split == 'train'

    if args.data_set.split('.')[0] == "geobench":
        transform = build_transform_geobench(is_train, args)
    else:
        transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set.split('.')[0] == "geobench":
        from geobenchdataset import GeobenchDataset
        # split = "train" if is_train else "val"
        dataset_name = args.data_set.split('.')[1]
        if dataset_name in ["m-eurosat", "m-so2sat", "m-bigearthnet", "m-brick-kiln"]:
            dataset = GeobenchDataset(dataset_name=dataset_name, split=split, transform=None, benchmark_name="classification")
        elif dataset_name in ["m-cashew-plantation", "m-SA-crop-type"]:
            dataset = GeobenchDataset(dataset_name=dataset_name, split=split, transform=None, benchmark_name="segmentation")
        else:
            raise NotImplementedError()
        nb_classes = dataset.num_classes
    else:
        raise NotImplementedError()

    if percent is None and num_samples is None:
        # set num_samples to a large number to avoid subsampling
        num_samples = 1000000000
    if (percent is not None and is_train) or (num_samples is not None and is_train):
        from geobenchdataset import geobench_dataset_subset
        if percent is not None:
            print("Subsampling the dataset to have only %d%% of the original samples" % (percent * 100))
        else:
            print("Subsampling the dataset to have only %d samples" % num_samples)
        # if args.data_set == 'geobench.m-bigearthnet':
        labels = []
        label_stats = dataset.label_stats
        label_map = dataset.label_map
        partition_stats = json.load(open(os.path.join(dataset.dataset_dir, "default_partition.json"), "r"))
        train_idx = partition_stats["train"]
        for i in train_idx:
            if label_stats is not None:
                label = np.where(np.array(label_stats[i]) == 1)[0]
            else:
                for k, v in label_map.items():
                    if i in v:
                        label = int(k)
                        break
            labels.append(label)
            
        print(len(labels))
        print('created labels')
        from subsample import stratified_subsample_multilabel
        if percent is not None:
            if args.data_set == 'geobench.m-bigearthnet':
                y = stratified_subsample_multilabel(labels, percentage=percent, multilabel=True, classes=[i for i in range(args.nb_classes)])
            else:
                y = stratified_subsample_multilabel(labels, percentage=percent, multilabel=False)
        else:
            if args.data_set == 'geobench.m-bigearthnet':
                if num_samples > len(labels):
                    num_samples = len(labels)
                y = stratified_subsample_multilabel(labels, num_samples=num_samples, multilabel=True, classes=[i for i in range(args.nb_classes)])
            else:
                if num_samples > len(labels):
                    num_samples = len(labels)
                y = stratified_subsample_multilabel(labels, num_samples=num_samples, multilabel=False)

        print('number of sub samples = ', len(y))
        if num_samples < len(labels):
            dataset = geobench_dataset_subset(dataset, y)
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

def build_transform_geobench(is_train, args):

    t = []
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
