# Multi Pretext Masked Autoencoder (MP-MAE)




[![Project Website](https://img.shields.io/badge/Project%20Website-8A2BE2)](https://vishalned.github.io/mmearth)
[![Paper](https://img.shields.io/badge/arXiv-2405.02771-blue)](https://arxiv.org/abs/2405.02771)
[![Code - Data](https://img.shields.io/badge/Code%20--%20Data-darkgreen)](https://github.com/vishalned/MMEarth-data/tree/main)


:bangbang:  **UPDATE: This repo is currently being fixed to work with the new v001 version of the data and with the ffcv dataloader. It will be ready by the first week of August 2024.** 

This repository contains code used to create the models and results presented in this paper [MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning](https://arxiv.org/abs/2405.02771). It modifies the [ConvNext V2](https://arxiv.org/abs/2301.00808) architecture to be used with [MMEarth](https://github.com/vishalned/MMEarth-data), which is a multi-modal geospatial remote sensing data. 

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2024.08.01 :fire::fire::fire:

- **Paper accepted to ECCV 2024 !!**
- Model now pretrained on MMEarth v001 & evaluated on GEO-Bench v1.0.
- Updated model scripts to work with MMEarth-v001.
- Data augmentation fix: Random crops are now aligned across modalities
- Test metrics fix: Metrics are now overall instead of mini-batch averages, matching GEO-Bench metrics.
- Added ffcv dataloader for both pretraining and finetuning. (training speed increased significantly.)


![model-grey](https://github.com/vishalned/MMEarth-train/assets/27778126/d7defca4-f603-4f00-af7d-f18e4fb3be84)

## Installation
See [INSTALL.md](https://github.com/vishalned/MMEarth-train/blob/main/INSTALL.md) for more instructions on the installation of dependencies


## Training 
See [TRAINING.md](https://github.com/vishalned/MMEarth-train/blob/main/TRAINING.md) for more details on training and finetuning. 

## Model Checkpoints
All the pretraining weights can be downloaded from [here](https://sid.erda.dk/sharelink/g23YOnaaTp). The folders are named in the format shown below. Inside the folder you will find a checkpoint `.pth` weight file. An example to load the weights is in the [examples](https://github.com/vishalned/MMEarth-train/tree/main/examples) folder.

```sh
CHECKPOINT FOLDER FORMAT
pt-($INPUT)_($MODEL)_($DATA)_($LOSS)_($MODEL_IMG_SIZE)_($PATCH_SIZE)/

$INPUT:
      - S2 # for s2-12 bands as input and output
      - all_mod # for s2-12 bands as input and all modalities as output
      - img_mod # for s2-12 bands as input and image level modalities as output
      - pix_mod # for s2-12 bands as input and pixel level modalities as output
      - rgb # for s2-bgr as input and output (we trained the model using bgr ordering)

$MODEL:
      - atto
      - tiny

$DATA:
      - 100k_128 # MMEarth100k, 100k locations and image size 128
      - 1M_64 # MMEarth64, 1.2M locations and image size 64
      - 1M_128 # MMEarth, 1.2M locations and image size 128

$LOSS: # loss weighting strategy
      - uncertainty
      - unweighted

$MODEL_IMG_SIZE # input size passed to the model
      - 56 # when using the data with image size 64
      - 112 # when using the data with image size 128

$PATCH_SIZE
      - 8
      - 16

Note: The only exception is when using the model trained on imagenet, the folder path is pt-imagenet_atto_200epochs_224_32/

```



## Acknowledgment
This repository borrows from the [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2/tree/main) repository.

## Citation
Please cite our paper if you use this code or any of the provided data.

Vishal Nedungadi, Ankit Kariryaa, Stefan Oehmcke, Serge Belongie, Christian Igel, & Nico Lang (2024). MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning.
```
@misc{nedungadi2024mmearth,
      title={MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning},
      author={Vishal Nedungadi and Ankit Kariryaa and Stefan Oehmcke and Serge Belongie and Christian Igel and Nico Lang},
      year={2024},
      eprint={2405.02771},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


