# Multi Pretext Masked Autoencoder (MP-MAE)




[![Project Website](https://img.shields.io/badge/Project%20Website-8A2BE2)](https://vishalned.github.io/mmearth)
[![Paper](https://img.shields.io/badge/arXiv-2405.02771-blue)](https://arxiv.org/abs/2405.02771)
[![Code - Data](https://img.shields.io/badge/Code%20--%20Data-darkgreen)](https://github.com/vishalned/MMEarth-data/tree/main)

This repository contains code used to create the models and results presented in this paper [MMEarth: Exploring Multi-Modal Pretext Tasks For Geospatial Representation Learning](https://arxiv.org/abs/2405.02771). It modifies the [ConvNext V2](https://arxiv.org/abs/2301.00808) architecture to be used with [MMEarth](https://github.com/vishalned/MMEarth-data), which is a multi-modal geospatial remote sensing data. 

![model-grey](https://github.com/vishalned/MMEarth-train/assets/27778126/d7defca4-f603-4f00-af7d-f18e4fb3be84)

## Installation
See [INSTALL.md](https://github.com/vishalned/MMEarth-train/blob/main/INSTALL.md) for more instructions on the installation of dependencies


## Training 
See [TRAINING.md](https://github.com/vishalned/MMEarth-train/blob/main/TRAINING.md) for more details on training and finetuning. 

## Model Checkpoints
All the pretraining weights can be downloaded from [here](https://sid.erda.dk/sharelink/ECYWkytzcG). The folders are named in the following format. Inside the folder you will find a checkpoint `.pth` weight file. An example to load the weights is in the [examples](https://github.com/vishalned/MMEarth-train/tree/main/examples) folder.

```sh
pt-all_mod_$MODEL_$DATA_$IMGSIZE_$LOSS/

$MODEL: atto or tiny
$DATA: 100k or 1M
$IMGSIZE: 128 or 64
$LOSS: uncertainty or unweighted # This is the loss weighting strategy. Most experiments in the paper were run using the uncertainty method. 
# note that while the img size is 128 or 64, during pretraining we use a random crop to make the image sizes 112 and 56 respectively. 
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


