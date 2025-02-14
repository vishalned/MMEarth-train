# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
import wandb
from geobench import GEO_BENCH_DIR
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_
from timm.utils import ModelEma

import helpers
import models.convnextv2 as convnextv2
import models.convnextv2_unet as convnextv2_unet
from custom_loss import LabelSmoothingBinaryCrossEntropy
# from datasets import build_dataset
from engine_finetune import train_one_epoch, evaluate
from geobenchdataset import get_geobench_dataloaders
from helpers import NativeScalerWithGradNormCount as NativeScaler
from helpers import str2bool, remap_checkpoint_keys, load_custom_checkpoint
from optim_factory import create_optimizer, LayerDecayValueAssigner

GEO_BENCH_DATASETS = ['m-eurosat', 'm-so2sat', 'm-bigearthnet', 'm-brick-kiln', 'm-cashew-plant', 'm-SA-crop-type']

def criterion_fn(args):
    '''
    Returns the criterion function based on the dataset


    TODO: Add more criterion functions for different datasets
    '''

    criterion_dict = {
        "m-eurosat": LabelSmoothingCrossEntropy(args.smoothing),
        "m-so2sat": LabelSmoothingCrossEntropy(args.smoothing),
        "m-bigearthnet": LabelSmoothingBinaryCrossEntropy(args.smoothing),
        "m-brick-kiln": LabelSmoothingCrossEntropy(args.smoothing),
        "m-cashew-plant": torch.nn.CrossEntropyLoss(),
        "m-SA-crop-type": torch.nn.CrossEntropyLoss(),
    }[args.data_set]

    return criterion_dict

def get_args_parser():
    parser = argparse.ArgumentParser("FCMAE fine-tuning", add_help=False)
    parser.add_argument("--batch_size", default=64, type=int, help="Per GPU batch size")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument(
        "--update_freq", default=1, type=int, help="gradient accumulation steps"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="convnextv2_base",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--input_size",
        default=112,
        type=int,
        help="image input size used for pretraining. This is useful since the patch size parameter depends on this",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--layer_decay_type",
        type=str,
        choices=["single", "group"],
        default="single",
        help="""Layer decay strategies. The single strategy assigns a distinct decaying value for each layer,
                        whereas the group strategy assigns the same decaying value for three consecutive layers""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="patch size used when pre-training the model",
    )
    parser.add_argument(
        "--baseline",
        type=str2bool,
        default=False,
        help="Whether to use the baseline model or not",
    )

    # EMA related parameters
    parser.add_argument("--model_ema", type=str2bool, default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument("--model_ema_force_cpu", type=str2bool, default=False, help="")
    parser.add_argument(
        "--model_ema_eval",
        type=str2bool,
        default=False,
        help="Using ema to eval during training.",
    )

    # Optimization parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument("--layer_decay", type=float, default=1.0)
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=20,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        type=str2bool,
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0.0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0.0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--head_init_scale",
        default=0.001,
        type=float,
        help="classifier head initial scale, typically adjusted in fine-tuning",
    )
    parser.add_argument(
        "--model_key",
        default="model|module",
        type=str,
        help="which key to load from saved state dict, usually model or model_ema",
    )
    parser.add_argument("--model_prefix", default="", type=str)

    # * Linear probe params
    parser.add_argument(
        "--linear_probe",
        type=str2bool,
        default=False,
        help="Whether to linear probe the model",
    )

    # Dataset parameters
    parser.add_argument(
        "--processed_dir",
        default=None,
        type=str,
        help="path to processed data (defaults to data location",
    )
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--eval_data_path", default=None, type=str, help="dataset path for evaluation"
    )
    parser.add_argument(
        "--data_set",
        default="m-bigearthnet",
        choices=[
            "m-eurosat",
            "m-so2sat",
            "m-bigearthnet",
            "m-brick-kiln",
            "m-cashew-plant",
            "m-SA-crop-type",
        ],
        type=str,
        help="Which dataset to use",
    )
    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_num", default=3, type=int)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--eval", type=str2bool, default=False, help="Perform evaluation only"
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument(
        "--pin_mem",
        type=str2bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Use apex AMP (Automatic Mixed Precision) or not",
    )

    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="mmearth-v001-ft")
    parser.add_argument("--wandb_run_name", type=str, default="")

    parser.add_argument("--pretraining", type=str, default="")
    parser.add_argument("--use_orig_stem", type=str2bool, default=False)
    parser.add_argument("--run_on_test", type=str2bool, default=False)
    parser.add_argument(
        "--partition",
        type=str,
        default="default",
        help="Amount of GeoBench data to train on. "
        'Available: "default", "0.01x_train", "0.02x_train", "0.05x_train", "0.10x_train", '
        '"0.20x_train", "0.50x_train", "1.00x_train" (default: "default").',
    )

    # a parameter to specify if we use geobench rgb, bgr, or full bands
    parser.add_argument(
        "--geobench_bands_type",
        type=str,
        default="full",
        choices=["full", "rgb", "bgr"],
        help="Type of bands to use for GeoBench dataset. "
        'Available: "full", "rgb", "bgr" (default: "full").',
    )
    
    parser.add_argument(
        "--use_imnet_weights",
        type=str2bool,
        default=False,
        help="Use ImageNet pretrained weights for the model",
    )
    parser.add_argument("--test_scores_dir", type=str, default="./test_scores/")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--version", type=str, default="1.0")
    parser.add_argument("--nb_classes", default=10, type=int)
    parser.add_argument("--no_ffcv", type=str2bool, default=False)
    parser.add_argument("--distributed", type=str2bool, default=False)
    return parser


def main(args: argparse.Namespace):
    # utils.init_distributed_mode(args)
    if args.distributed:
        helpers.init_distributed_mode(args)

    print(args)
    device = torch.device(args.device)

    if '.pth' not in args.finetune and '.pt' not in args.finetune:
        # it is a directory, so we get the last checkpoint
        args.finetune = os.path.join(args.finetune, sorted(os.listdir(args.finetune))[-1])

    # fix the seed for reproducibility
    seed = args.seed + helpers.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    log_writer = None

    processed_dir = args.processed_dir
    if args.data_set in GEO_BENCH_DATASETS:
        if processed_dir is None:
            processed_dir = GEO_BENCH_DIR
        (train_dataloader, val_dataloader), task = get_geobench_dataloaders(
            args.data_set,
            processed_dir,
            args.num_workers,
            args.batch_size,
            ["train", "val"],
            args.partition,
            indices=[list(range(10)), list(range(10))] if args.debug else None,
            version=args.version,
            geobench_bands_type=args.geobench_bands_type,
            no_ffcv=args.no_ffcv,
            seed=args.seed,
        )
    else:
        # call a new dataset function here
        # TODO: Add new dataset function here
        raise ValueError(f"Unknown dataset: {args.data_set}") 
    
    
    num_classes = task.num_classes
    samples, targets, _, _ = next(iter(train_dataloader))
    in_channels = samples.shape[1]
    print('in_channels:', in_channels)
    print('num_classes:', num_classes)
    args.nb_classes = num_classes

 ############################## LOADING MODEL AND FREEZING/UNFREEING MODEL #############################
    # resnet is used for other benchmarking models. For MMEarth, we use convnextv2 models
    if 'resnet' in args.model:
        if 'unet' in args.model:
            import segmentation_models_pytorch as smp
            model_name = 'resnet18' if '18' in args.model else 'resnet50'
            model = smp.Unet(
                encoder_name=model_name,
                encoder_weights=None,
                in_channels=in_channels,
                classes=args.nb_classes
            )
        else:
            model = torchvision.models.__dict__[args.model](pretrained=False)
    else:
        # convnextv2 unet models are also loaded here
        model = convnextv2.__dict__[args.model](
                num_classes=num_classes,
                drop_path_rate=args.drop_path,
                head_init_scale=args.head_init_scale,
                args=args,
                patch_size=args.patch_size,
                img_size=args.input_size,
                use_orig_stem=args.use_orig_stem,
                in_chans=in_channels,
            )

    model, _ = load_custom_checkpoint(model, args) # freezing and unfreezing is done in this function
    model.to(device)


    model_without_ddp = model
    if 'resnet50' in args.model:
        model_without_ddp.depths = [3, 4, 6, 3]
    elif 'resnet18' in args.model:
        model_without_ddp.depths = [2, 2, 2, 2]


    ####################################################################################################

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)
    eff_batch_size = args.batch_size * args.update_freq * helpers.get_world_size()
    num_training_steps_per_epoch = len(train_dataloader) // eff_batch_size
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        assert args.layer_decay_type in ["single", "group"]
        if args.layer_decay_type == "group":  # applies for Base and Large models
            num_layers = 12
        else:
            num_layers = sum(model_without_ddp.depths)
        assigner = LayerDecayValueAssigner(
            list(
                args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            ),
            depths=model_without_ddp.depths,
            layer_decay_type=args.layer_decay_type,
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module



    ############################## OPTIMIZER and LOSS SCALER #####################
    optimizer = create_optimizer(
        args,
        model_without_ddp,
        skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None,
    )
    loss_scaler = NativeScaler(args.device)
    ##############################################################################




    ############################## LOSS FUNCTION #############################
    criterion = criterion_fn(args)
    print("Criterion = %s" % str(criterion))
    ##########################################################################



    

    helpers.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=None,
    )
        


    ############################## MAIN TRAINING LOOP #############################

    max_accuracy = 0.0
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if "segmentation" in task.label_type:
            # for unet we probe the decoder only for 50 epochs, and then fine-tune for 150 epochs
            if epoch == 50:
                if "resnet" in args.model:
                    print("unfreezing the encoder part of the model")
                    for param in model_without_ddp.encoder.parameters():
                        param.requires_grad = True

                    optimizer.add_param_group({"params": model_without_ddp.encoder.parameters()})
                else:
                    print("Unfreezing the encoder part of the model")
                    for param in model_without_ddp.parameters():
                        param.requires_grad = True

                    new_param_groups = [
                        {"params": model_without_ddp.downsample_layers.parameters()},
                        {"params": model_without_ddp.stages.parameters()},
                        {"params": model_without_ddp.initial_conv.parameters()},
                        {"params": model_without_ddp.stem.parameters()}
                    ]

                    optimizer.add_param_group({"params": [p for group in new_param_groups for p in group["params"]]})
        
        train_stats = train_one_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            None, #model_ema
            None, #mixup
            log_writer=log_writer,
            args=args,
            task=task,
        )
    
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                helpers.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=None,
                )
        
        # validation        
        if val_dataloader is not None:
            try:
                val_samples = val_dataloader.reader.num_samples
            except:
                val_samples = len(val_dataloader.dataset)

            test_stats = evaluate(
                val_dataloader, model, device, use_amp=args.use_amp, args=args, task=task
            )

            logging_text = "Metric: "
            for k, v in train_stats.items():
                if k != "loss":
                    logging_text += f"{k} - {v:.3f} "
                    metric_key = k
            logging_text += f" on {val_samples} test samples"   

            if max_accuracy < test_stats[metric_key]:
                max_accuracy = test_stats[metric_key]
                if args.output_dir and args.save_ckpt:
                    helpers.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=None,
                    )
                print(f"Max accuracy: {max_accuracy:.2f}%")

    

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.wandb:
                wandb.log(log_stats)

        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir and helpers.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
    ###############################################################################





    ################################ RUN ON TEST SET ################################
    if args.run_on_test:
        # use the last model
        if args.distributed:
            args.local_rank = 0
            args.distributed = False
        
        if "segmentation" not in task.label_type:
            ckpt_file = 'checkpoint-99.pth'
        else:
            ckpt_file = 'checkpoint-199.pth'

        checkpoint = torch.load(
            os.path.join(args.output_dir, ckpt_file), map_location="cpu"
        )
        print(
            "Load pre-trained checkpoint from: %s"
            % os.path.join(args.output_dir, ckpt_file)
        )

        # load the model directly with the checkpoint
        helpers.load_state_dict(model, checkpoint["model"], prefix=args.model_prefix)
        model.to(device)
        if args.data_set in GEO_BENCH_DATASETS:
            (test_loader,), task = get_geobench_dataloaders(
                args.data_set,
                processed_dir,
                args.num_workers,
                1, # batch size
                ["test"],
                args.partition,
                indices=[list(range(10))] if args.debug else None,
                version=args.version,
                geobench_bands_type=args.geobench_bands_type,
                no_ffcv=args.no_ffcv,
                
            )
        else:
            raise ValueError(f"Unknown dataset: {args.data_set}")

        print('test_loader data shape:', next(iter(test_loader))[0].shape)

        test_stats = evaluate(
            test_loader, model, device, use_amp=args.use_amp, args=args, task=task
        )
        test_samples = test_loader.reader.num_samples
        key = [k for k in test_stats.keys() if k != 'loss'][0]
        print(f"Final test set - {test_samples} samples, score: {test_stats[key]:.3f}")
        test_score = test_stats[key]
        
        ## some code for logging in a text file
        # if helpers.is_main_process():
        #     if (
        #         task.label_type.__class__ == SegmentationClasses
        #     ):
        #         file_str = f"unet_lp&ft--{args.data_set}--{args.pretraining}.txt"
        #     else:
        #         if args.partition in ["default", "1.00x_train"]:
        #             file_str = f"{'lp' if args.linear_probe else 'ft'}--{args.data_set}--{args.pretraining}.txt"
        #         else:
        #             text = args.partition
        #             file_str = f"{'lp' if args.linear_probe else 'ft'}--{args.data_set}--{args.pretraining}--{text}.txt"

        #     if not os.path.exists(args.test_scores_dir):
        #         os.makedirs(args.test_scores_dir, exist_ok=True)

        #     with open(
        #         os.path.join(args.test_scores_dir, file_str), mode="a", encoding="utf-8"
        #     ) as f:
        #         write_str = f"test score: {test_score}, val_score: {max_accuracy}\n"
        #         f.write(write_str)

    wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("FCMAE fine-tuning", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.wandb:
        print("Logging to wandb")
        config = {
            "pretraining": args.pretraining,
            "data_set": args.data_set,
            "linear_probe": args.linear_probe,
        }
        wandb.init(project=args.wandb_project, config=config)
        wandb.run.name = args.wandb_run_name

    main(args)
