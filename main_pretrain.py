# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import numpy as np
import time
import json
import os
import uuid
from pathlib import Path


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from engine_pretrain import train_one_epoch
from custom_dataset import build_pretraining_dataset
from custom_loss import UncertaintyWeightingStrategy
import models.fcmae as fcmae

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool
from MODALITIES import *
import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE pre-training', add_help=False)

    # wandb parameters
    parser.add_argument('--wandb', type=str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default='global-lr')
    parser.add_argument('--wandb_run_name', type=str)




    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation step')
    
    

    # loss parameters
    parser.add_argument('--loss_aggr', choices=['uncertainty', 'unweighted'], default='uncertainty', 
                        help='loss aggregation method')
    parser.add_argument('--loss_full', type=str2bool, default='False', 
                        help='compute loss on all pixels or only on masked pixels') # true means compute loss on all pixels
    
    
    # Model parameters
    parser.add_argument('--model', default='convnextv2_pico', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=112, type=int,
                        help='image input size')
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', type=str2bool, default=False,
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--use_orig_stem', type=str2bool, default=False)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new.h5', type=str,
                        help='path to the h5 file')
    parser.add_argument('--random_crop', type=str2bool, default=True)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # modality parameters
    parser.add_argument('--mod_setting', choices=['s2_only', 's2_rgb', 'None', 'full'], default='None', 
                        help='Modality input, output setting. Used for slurm execution')
    
    return parser



def main(args):

    utils.init_distributed_mode(args)
    # args.distributed = False
    print(args)

    ############# creating some additional args variables to be used by other functions #############
    if args.mod_setting != "None":
        args.inp_modalities = MOD_DICT[args.mod_setting]["INP_MODALITIES"]
        args.out_modalities = MOD_DICT[args.mod_setting]["OUT_MODALITIES"]
    else:
        args.inp_modalities = INP_MODALITIES
        args.out_modalities = OUT_MODALITIES
        
    args.modalities = args.inp_modalities.copy()
    args.modalities.update(args.out_modalities)

    args.modalities_full = MODALITIES_FULL
    if not args.IMNET:
        args.data_name = args.data_path.split('.')[0].split('/')[-1]
        args.splits_path = args.data_path.split('.')[0] + '_splits.json'
        args.tile_info_path = args.data_path.split('.')[0] + '_tile_info.json'
        args.band_stats_path = args.data_path.split('.')[0] + '_band_stats.json'

        # incase needed - hardcoding the paths
        # args.data_name = 'data_1M_130_new'
        # args.splits_path = '/projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new_splits.json'
        # args.tile_info_path = '/projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new_tile_info.json'
        # args.band_stats_path = '/projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new_band_stats.json'

        # quick check to see if all the files exist
        assert os.path.exists(args.data_path), "Data file does not exist"
        assert os.path.exists(args.splits_path), "Split file does not exist"
        assert os.path.exists(args.tile_info_path), "Tile info file does not exist"
        assert os.path.exists(args.band_stats_path), "Band stats file does not exist"

        args.band_stats = json.load(open(args.band_stats_path, 'r'))
        args.tile_info = json.load(open(args.tile_info_path, 'r'))

    
    #################################################################################################

    if args.wandb and args.local_rank == 0:
        print("Logging to wandb")
        config = {
            'model': args.model ,
            'mask_ratio': args.mask_ratio,
            'norm_pix_loss': args.norm_pix_loss,
            # 'loss_type': args.loss_type,
            'loss_aggr': args.loss_aggr,
            'loss_full': args.loss_full,
            'patch_size': args.patch_size,
            'input_size': args.input_size,
            'blr': args.blr,
            'batch_size': args.batch_size,
            'update_freq': args.update_freq,
            'use_orig_stem': args.use_orig_stem
        }
    
        wandb.init(project=args.wandb_project, config=config)
        wandb.run.name = args.wandb_run_name


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    

    # custom multimodal dataset
    dataset_train = build_pretraining_dataset(is_train=True, args=args)


    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    
    def collate_fn(batch):
        # for each batch append the samples of the same modality together and return the ids. We keep track of the ids to differentiate between sentinel2_l1c and sentinel2_l2a
        return_batch = {}
        ids = [b['id'] for b in batch]
        return_batch = {modality: torch.stack([b[modality] for b in batch], dim=0) for modality in args.modalities.keys()}
        return ids, return_batch

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_fn if not args.IMNET else None,
    )

    if args.loss_aggr == 'uncertainty':
        num_tasks = len(args.out_modalities) # in this case we have one uncertainty value per modality
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
        loss_fn=loss_fn
    )
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print('number of params in encoder:', n_parameters_encoder)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module



    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)


######## Training loop ########
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_loss = 100000
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats, loss_dict, log_var_list, normalized_loss_list = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and args.save_ckpt:
            if (epoch+1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

    
        if args.wandb and utils.is_main_process():
            # we also log multiple loss values and log_var values if its not None
            loss_dict_keys = []
            for k, v in loss_dict.items():
                log_stats[f'train_{k}'] = v
                loss_dict_keys.append(k)
            if log_var_list is not None:
                for i, v in enumerate(log_var_list):
                    mod = loss_dict_keys[i]
                    log_stats[f'log_var_{mod}'] = v
            if normalized_loss_list is not None:
                for i, v in enumerate(normalized_loss_list):
                    mod = loss_dict_keys[i]
                    log_stats[f'normalized_loss_{mod}'] = v

            wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    main(args)
