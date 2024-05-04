# Multi Pretext Masked Autoencoder (MP-MAE)
![model-grey](https://github.com/vishalned/MMEarth-train/assets/27778126/b4461d84-0f42-4489-acec-77834335be94)


[![Project Website](https://img.shields.io/badge/Project%20Website-8A2BE2)](https://vishalned.github.io/mmearth)
[![Paper](https://img.shields.io/badge/arXiv-xxxx.xxxxx-blue)](https://arxiv.org/abs/xxxx.xxxxx)
[![Code - Data](https://img.shields.io/badge/Code%20--%20Data-darkgreen)](https://github.com/vishalned/MMEarth-data/tree/main)

This repository contains code used to create the models and results presented in this paper [MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning](). It modifies the [ConvNext V2](https://arxiv.org/abs/2301.00808) architecture to be used with [MMEarth](https://github.com/vishalned/MMEarth-data), which is a multi-modal geospatial remote sensing data. 


## Installation
See [INSTALL.md](https://github.com/vishalned/MMEarth-train/blob/main/INSTALL.md) for more instructions on the installation of dependencies


## Training 
See [TRAINING.md](https://github.com/vishalned/MMEarth-train/blob/main/TRAINING.md) for more details on training and finetuning. 

## Model Checkpoints
All the pretraining weights can be downloaded from [here](https://sid.erda.dk/sharelink/ECYWkytzcG). The folders are named in the following format. Inside the folder you will find a checkpoint `.pth` weight file. 

```sh
pt-all_mod_$MODEL_$DATA_$IMGSIZE_$LOSS/

$MODEL: atto or tiny
$DATA: 100k or 1M
$IMGSIZE: 128 or 64
$LOSS: uncertainty or unweighted # This is the loss weighting strategy. Most experiments in the paper were run using the uncertainty method. 
```

## Acknowledgment
This repository borrows from the [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2/tree/main) repository.

## Citation
Please cite our paper if you use this code or any of the provided data.

Vishal Nedungadi, Ankit Kariryaa, Stefan Oehmcke, Serge Belongie, Christian Igel, & Nico Lang (2024). MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning.
```
  @article{mmearth2024,
    title={{MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning}},
    author={Vishal Nedungadi and Ankit Kariryaa and Stefan Oehmcke and Serge Belongie and Christian Igel and Nico Lang},
    year={2024},
}
```


