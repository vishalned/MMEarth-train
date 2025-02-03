# Installation

Most installation steps are similar to [ConvNeXtV2](https://github.com/facebookresearch/ConvNeXt-V2), and we provide the instructions below. 

To make the installation easier, you can either choose to manually install the packages as shown below, or use the `env.yml` file and install it using mamba `mamba env create -f env.yml`. 

## MP-MAE Pretraining (sparse convolutions)

This installation is tested for CUDA 11.8.

Creating a new conda environment
```sh
conda create -n mmearth-train python=3.9 -y
conda activate mmearth-train

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Possible errors:** If you notice any errors regarding libnccl.so.2 when importing torch, a solution that might work is described [here](https://stackoverflow.com/questions/75879951/torch-2-installed-could-not-load-library-libcudnn-cnn-infer-so-8-error-libnv
). TLDR: you need to update the LD_LIBRARY_PATH environment variable to the correct cuda libnccl.so.2 path.

Install Minkoswki Engine (this is only required for pre-training the model from scratch, since ConvNeXt V2 uses sparse convolutions, which are implemented in the Minkowski Engine):

We use GCC 11.X for the installation.

```sh
git submodule update --init --recursive
git submodule update --recursive --remote
conda install openblas-devel -c anaconda

cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

**Possible errors:** Incase you get errors referencing the `at:Tensor` variable in `src/spmm.cu` file. Consider adding `#include <ATen/core/Tensor.h>` to the same file and re run the setup. Make sure the added line is above `#include <ATen/cuda/CUDAContext.h>`.

If you want to run the code using ffcv, you also need to install this:

```sh
conda config --env --set channel_priority flexible
conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c conda-forge
pip install ffcv
```
**Possible errors:** If you see an error with ffcv installations, first uninstall ffcv and reinstall using `pip install ffcv --no-cache-dir`.


## MP-MAE Finetuning

Creating a new conda environment
```sh
conda create -n mmearth-train python=3.9 -y
conda activate mmearth-train

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

If you want to run the code using ffcv, you also need to install this:

```sh
conda config --env --set channel_priority flexible
conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c conda-forge
pip install ffcv
```

Install [GEO-Bench](https://github.com/ServiceNow/geo-bench) for finetuning:
```sh
pip install geobench
```
Visit their website for the complete installation and data downloading guide.
