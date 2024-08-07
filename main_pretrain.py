# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import json
import time

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import wandb

import helpers
import models.fcmae as fcmae
from MODALITIES import *
from custom_loss import UncertaintyWeightingStrategy
from engine_pretrain import train_one_epoch
from helpers import NativeScalerWithGradNormCount as NativeScaler
from helpers import str2bool
from mmearth_dataset import get_mmearth_dataloaders


def get_args_parser():
    parser = argparse.ArgumentParser("FCMAE pre-training", add_help=False)

    # wandb parameters
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="global-lr")
    parser.add_argument("--wandb_run_name", type=str)

    parser.add_argument("--batch_size", default=64, type=int, help="Per GPU batch size")
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument(
        "--update_freq", default=1, type=int, help="gradient accumulation step"
    )

    # loss parameters
    parser.add_argument(
        "--loss_aggr",
        choices=["uncertainty", "unweighted"],
        default="uncertainty",
        help="loss aggregation method",
    )
    parser.add_argument(
        "--loss_full",
        type=str2bool,
        default="False",
        help="compute loss on all pixels or only on masked pixels",
    )  # true means compute loss on all pixels

    # Model parameters
    parser.add_argument(
        "--model",
        default="convnextv2_pico",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=112, type=int, help="image input size")
    parser.add_argument(
        "--mask_ratio",
        default=0.6,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )
    parser.add_argument(
        "--norm_pix_loss",
        type=str2bool,
        default=False,
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.add_argument("--decoder_depth", type=int, default=1)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--use_orig_stem", type=str2bool, default=False)

    # Optimizer parameters
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
        default=1.5e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_dir", default=MMEARTH_DIR, type=str, help="path to the dataset"
    )
    parser.add_argument(
        "--processed_dir",
        default=None,
        type=str,
        help="path to the processed dataset (beton file)",
    )
    parser.add_argument("--random_crop", type=str2bool, default=True)
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_num", default=3, type=int)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--use_mixed", type=str2bool, default=False)
    parser.add_argument("--sparse", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--no_ffcv", type=str2bool, default=True)

    return parser


def main(args):


    if args.distributed:
        helpers.init_distributed_mode(args)
    print(args)

    args.data_dir = Path(args.data_dir)

    ############# creating some additional args variables to be used by other functions #############
    args.inp_modalities = INP_MODALITIES
    args.out_modalities = OUT_MODALITIES
    args.modalities = args.inp_modalities.copy()
    args.modalities.update(args.out_modalities)

    args.modalities_full = MODALITIES_FULL
    #################################################################################################

    if (args.wandb and args.local_rank == 0) or (args.wandb and args.distributed == False):
        print("Logging to wandb")
        config = {
            "model": args.model,
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": args.norm_pix_loss,
            # "loss_type": args.loss_type,
            "loss_aggr": args.loss_aggr,
            "loss_full": args.loss_full,
            "patch_size": args.patch_size,
            "input_size": args.input_size,
            "blr": args.blr,
            "batch_size": args.batch_size,
            "update_freq": args.update_freq,
            "use_orig_stem": args.use_orig_stem,
        }

        wandb.init(project=args.wandb_project, config=config)
        wandb.run.name = args.wandb_run_name

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + helpers.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    num_tasks = helpers.get_world_size()
    global_rank = helpers.get_rank()

    # custom multimodal dataset
    if not args.no_ffcv:
        train_dataloader = get_mmearth_dataloaders(
            args.data_dir,
            args.processed_dir,
            args.modalities,
            num_workers=args.num_workers,
            batch_size_per_device=args.batch_size,
            distributed=args.distributed,
            indices=(
                [np.arange(10)] if args.debug else None
            ),  # only 10 samples if debug is enabled
        )[0]
        len_dataset = train_dataloader.reader.num_samples
    else:
        def collate_fn(batch):
        # for each batch append the samples of the same modality together and return the ids. We keep track of the ids to differentiate between sentinel2_l1c and sentinel2_l2a
            return_batch = {}
            ids = [b['id'] for b in batch]
            return_batch = {modality: torch.stack([torch.from_numpy(b[modality]) for b in batch], dim=0) for modality in args.modalities.keys()}
            return ids, return_batch
        
        dataset = get_mmearth_dataloaders(
            args.data_dir,
            args.processed_dir,
            args.modalities,
            num_workers=args.num_workers,
            batch_size_per_device=args.batch_size,
            distributed=args.distributed,
            no_ffcv=args.no_ffcv,
        )[0] # non ffcv mode returns only the dataset object


        sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        len_dataset = len(dataset)
    
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        log_writer = helpers.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.loss_aggr == "uncertainty":
        num_tasks = len(
            args.out_modalities
        )  # in this case we have one uncertainty value per modalities
        loss_fn = UncertaintyWeightingStrategy(num_tasks)
    else:
        loss_fn = None
    # define the model
    model = fcmae.__dict__[args.model](
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        decoder_embed_dim=args.decoder_embed_dim,
        norm_pix_loss=args.norm_pix_loss,
        patch_size=args.patch_size,
        img_size=args.input_size,
        args=args,
        loss_fn=loss_fn,
        sparse=args.sparse,
    )
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_encoder = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)
    print("number of params in encoder:", n_parameters_encoder)

    eff_batch_size = args.batch_size * args.update_freq * helpers.get_world_size()
    num_training_steps_per_epoch = len_dataset // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    try:
        param_groups = optim_factory.add_weight_decay(
            model_without_ddp, args.weight_decay
        )
    except AttributeError:  # newer version do this
        param_groups = optim_factory.param_groups_weight_decay(
            model_without_ddp, args.weight_decay
        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(args.device)

    helpers.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    ######## Training loop ########
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # not needed for ffcv? we do not recompile
        if args.distributed and args.no_ffcv:
            train_dataloader.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats, loss_dict, log_var_list, normalized_loss_list = train_one_epoch(
            model,
            args.modalities,
            train_dataloader,
            optimizer,
            device,
            epoch,
            args.use_mixed,
            loss_scaler,
            log_writer=log_writer,
            args=args,
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
                )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if (args.wandb and helpers.is_main_process()) or (args.wandb and args.distributed == False):
            # we also log multiple loss values and log_var values if its not None
            loss_dict_keys = []
            for k, v in loss_dict.items():
                log_stats[f"train_{k}"] = v
                loss_dict_keys.append(k)
            if log_var_list is not None:
                for i, v in enumerate(log_var_list):
                    mod = loss_dict_keys[i]
                    log_stats[f"log_var_{mod}"] = v
            if normalized_loss_list is not None:
                for i, v in enumerate(normalized_loss_list):
                    mod = loss_dict_keys[i]
                    log_stats[f"normalized_loss_{mod}"] = v

            wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    wandb.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
