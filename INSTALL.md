# Installation

Most installation steps are similar to [ConvNeXtV2](https://github.com/facebookresearch/ConvNeXt-V2), and we provide the instructions below. 

## MP-MAE Pretraining (sparse convolutions)

Creating a new conda environment
```sh
conda create -n convnextv2 python=3.9 -y
conda activate convnextv2
pip install -r requirements.txt
```

Install Minkoswki Engine (this is only required for pre-training the model from scratch, since ConvNeXt V2 uses sparse convolutions, which are implemented in the Minkowski Engine):
(Note: Incase of any issues, follow the readme and issues in the official [repo](https://github.com/shwoo93/MinkowskiEngine/tree/bbc30ef581ea6deb505976b663f5fc2358a83749)).

```sh
git submodule update --init --recursive
git submodule update --recursive --remote
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```


## MP-MAE Finetuning

Creating a new conda environment
```sh
conda create -n convnextv2 python=3.9 -y
conda activate convnextv2
pip install -r requirements.txt
```

Install [GEO-Bench](https://github.com/ServiceNow/geo-bench) for finetuning:
```sh
pip install geobench
```
Visit their website for the complete installation and data downloading guide.