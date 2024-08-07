# Training
For all pre-training, we maintain an effective batch size of 4096. This is to ensure that we can use similar hyperparameters that were used by ConvNeXt V2. Effective batch size is `batch_size * update_freq * num_of_gpus`.

## Pretraining

**Arguments:** Most arguments are similar to the ones in ConvNeXt V2 repo, but here are some additional ones added. 

```sh
--loss_aggr # loss aggregation strategy to account for multiple losses for each modality. We have implemented the uncertainty method (as explained by Kendall et al.) or the unweighted method (equal weighting of all losses). 
--random_crop # enables random cropping the image to a size of (input_size x input_size). This works well when the input_size is 112 or 58 (given that MMEarth data is of size 128 or 64). 
--use_orig_stem # uses the original stem as in ConvNeXt V2, or a modified version of the stem. For our experiments we use the modified stem.
--no_ffcv # a boolean variable that indicates if you want to use ffcv or not
```

**Modalities:** All modalities required for training must be modified in the `MODALITIES.py` file. The comments in that file will provide some more instructions on what to change.


**Pretraining an atto model using 8 gpus**:
```sh
python  -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
        --model convnextv2_atto \
        --batch_size 256 \
        --update_freq 2 \
        --blr 1.5e-4 \
        --epochs 200 \
        --warmup_epochs 40 \
        --data_dir /projects/dereeco/data/global-lr/data_1M_130_new \
        --output_dir /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-all_mod_atto \
        --wandb True \
        --wandb_run_name sanity_check_atto \
        --wandb_project global-lr \
        --loss_aggr uncertainty \
        --auto_resume False \
        --norm_pix_loss True \
        --num_workers 8 \
        --patch_size 16 \
        --input_size 112 \
        --random_crop True \
        --use_orig_stem False \
        --save_ckpt True \
        --no_ffcv True
```


## Finetuning
**Arguments:** Most arguments are similar to the ones in ConvNeXt V2 repo, but here are some additional ones added. 

```sh

--use_orig_stem # uses the original stem as in ConvNeXt V2, or a modified version of the stem. For our experiments we use the modified stem.
--patch_size # ensure this is the same as what was used for the pretrained model.
--input_size # ensure this is also the same as what was used for the pretrained model. NOTE: This is not the image size of the finetuning dataset.
--no_ffcv # a boolean variable that indicates if you want to use ffcv or not
--run_on_test # a boolean variable that indicates if you want to report the metrics on the test set or not
--geobench_bands_type # 'bgr' or 'full' or 'rgb'. this indicates if you want to use all the bands, bgr bands or rgb bands in the geobench dataset.
--partition # ("0.01x_train" "0.05x_train" "0.50x_train", "default") what percentage subset of the geobench dataset to use.
```

**Bands:** We have enabled finetuning on GEO-Bench datasets (bigearthnet, so2sat, eurosat, cashew-plantation, SA-crop-type, brick-kiln). The datasets are named like `m-bigearthnet` for example. Modify `BAND_NAMES.json`, to change the bands that need to be used for these datasets. You can either use 3 bands (RGB) or all 12 bands. (eg. In the case of so2sat, the dataset doesnt consist of 12 bands, hence we replace the missing bands with ones containing similar wavelength (B1 -> B2, B9 -> B8A)).

**Finetuning on GEO-Bench on a single gpu:**
```sh
python -m  main_finetune \
            --model convnextv2_atto \
            --batch_size 32 \
            --update_freq 2 \
            --blr 2e-4 \
            --epochs 200 \
            --warmup_epochs 0 \
            --layer_decay_type 'single' \
            --layer_decay 0.9 \
            --weight_decay 0.3 \
            --drop_path 0.1 \
            --reprob 0.25 \
            --mixup 0. \
            --cutmix 0. \
            --smoothing 0.2 \
            --finetune /projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-all_mod_uncertainty/checkpoint-199.pth \
            --output_dir "/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/testing" \
            --data_set "m-bigearthnet" \
            --linear_probe True \
            --pretraining testing \
            --wandb False \
            --auto_resume False \
            --patch_size 16 \
            --input_size 112 \
            --use_orig_stem False \
            --save_ckpt False \
            --no_ffcv False \
            --run_on_test True \
            --num_workers 2 \
            --geobench_bands_type 'bgr' \
            --partition 'default' 
```

**Note:** In the case of the segmentations datasets (cashew-plantation and SA-crop-type), you need to change the model and smoothing argument as follows:
```sh
--model convnextv2_unet_atto \
--smoothing 0 \ # ensure smoothing is 0.
```

## Slurm Scripts
For all our training, we use slurm, and have the scripts [here](https://github.com/vishalned/MMEarth-train/tree/main/slurm_scripts). Comments inside the slurm scripts will give you better insights on what each script was used for. 
