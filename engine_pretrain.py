# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
import sys
from typing import List, Dict, AnyStr

import ffcv
import torch

import helpers


def train_one_epoch(
    model: torch.nn.Module,
    modalities: Dict[AnyStr,List],
    data_loader: ffcv.Loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train()
    metric_logger = helpers.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", helpers.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    update_freq = args.update_freq

    optimizer.zero_grad()
    for data_iter_step, data in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        samples = helpers.make_modality_dict(data, modalities)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            helpers.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        if not isinstance(samples, list):
            if isinstance(samples, dict):
                for key in samples:
                    samples[key] = samples[key].to(device, non_blocking=True)

        loss, pred, mask, loss_dict_, log_var_list_, normalized_loss_list_ = model(
            samples, mask_ratio=args.mask_ratio
        )
        # loss_dict_ is a dictionary containing the loss values for each modalities
        # log_var_list_ is a list containing the log variance for each modality - used in uncertainty weighting
        # normalized_loss_list_ is a list containing the normalized loss for each modality - used in uncertainty weighting

        log_var_list = log_var_list_
        normalized_loss_list = (
            normalized_loss_list_.detach().cpu().numpy()
            if normalized_loss_list_ is not None
            else None
        )

        loss_value = loss.item()
        loss_dict = {}
        for k, v in loss_dict_.items():
            loss_dict[k] = v.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= update_freq

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % update_freq == 0,
        )
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # clear the GPU cache at a regular interval for training ME network

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = helpers.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.update(
                train_loss=loss_value_reduce, head="loss", step=epoch_1000x
            )
            log_writer.update(lr=lr, head="opt", step=epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        loss_dict,
        log_var_list,
        normalized_loss_list,
    )
