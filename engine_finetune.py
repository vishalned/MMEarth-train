# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

try:
    import ffcv
except:
    print("ffcv not installed")

import torch
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import accuracy, ModelEma
import torchmetrics
from geobench.dataset import SegmentationClasses
from geobench.label import Classification, MultiLabelClassification
from geobench.task import TaskSpecifications
from typing import Callable, Dict, List, Optional, Union

import helpers
from helpers import adjust_learning_rate

def eval_metrics_generator(task_specs) -> List[torchmetrics.MetricCollection]:
    """Return the appropriate eval function depending on the task_specs.

    Args:
        task_specs: a dictionary containing the task specifications

    Returns:
        metric collection used during evaluation
    """

    if task_specs.num_classes == 1:
        metrics: List[torchmetrics.MetricCollection] = {
            "agbd": torchmetrics.MetricCollection({"MSE": helpers.CustomMSE()}),
        }[task_specs.dataset]
    else:
        metrics: List[torchmetrics.MetricCollection] = {
            "m-eurosat": torchmetrics.MetricCollection({"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=task_specs.num_classes, average="micro")}),
            "m-cashew-plant": torchmetrics.MetricCollection({"Jaccard": torchmetrics.JaccardIndex(task="multiclass", num_classes=task_specs.num_classes, average="macro")}),
            "m-SA-crop-type": torchmetrics.MetricCollection({"Jaccard": torchmetrics.JaccardIndex(task="multiclass", num_classes=task_specs.num_classes, average="macro")}),
            "m-bigearthnet": torchmetrics.MetricCollection({"F1Score": torchmetrics.F1Score(task="multilabel", num_labels=task_specs.num_classes, average="micro")}),
            "m-so2sat": torchmetrics.MetricCollection({"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=task_specs.num_classes, average="micro")}),
            "m-brick-kiln": torchmetrics.MetricCollection({"Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=task_specs.num_classes, average="micro")}),
    }[task_specs.dataset]

    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: ffcv.Loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
    task=None
):
    model.train()
    metric_logger = helpers.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", helpers.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    metric = eval_metrics_generator(task).to(device)
    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        if args.use_imnet_weights:
            # since we are making use of the weights trained on imagenet, we need to ensure the geobench is rgb. Hence if it is bgr, we reaarange the channels
            if args.geobench_bands_type == "bgr":
                samples = samples[:, [2, 1, 0], :, :]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            output = model(samples)
            if task.label_type == "segmentation":
                # make output class the last dimension
                output = output.permute(0, 2, 3, 1)
                # assuming for segmentation we have the cross entropy loss, we need to convert the output to something of shape N, C
                output_tmp = output.contiguous().view(-1, output.size(3))
                targets = targets.unsqueeze(1) if args.no_ffcv else targets

                target_tmp = targets.permute(0, 2, 3, 1)
                target_tmp = target_tmp.contiguous().view(-1, target_tmp.size(3))
                target_tmp = target_tmp.squeeze(1) # cross entropy loss expects a 1D tensor
                target_tmp = target_tmp.long()
            else:
                target_tmp = targets
                output_tmp = output

            loss = criterion(output_tmp, target_tmp)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            # loss.requires_grad = True
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if device.__str__ == "cuda":
            torch.cuda.synchronize()

        if task.label_type == "segmentation":
            # for segmentation we calculate the mean intersection over union, hence the jaccard index
            output = output.permute(0, 3, 1, 2) # N, C, H, W
            output = torch.nn.functional.softmax(output, dim=1) # argmax already applied in the metric
            targets = targets.squeeze(1)


        score = metric(output, targets) # for bigearthnet, sigmoid is already applied in the metric

        metric_logger.update(loss=loss_value)
        # metric_logger.update(score=score)
        for key in score.keys():
            metric_logger.meters[key].update(score[key].item())


        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)

    # we create a dict, with the metrics overwritten with metric.copute() values, and the rest as the global average
    metric_values = metric.compute()
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
    for key in metric_values.keys(): # overwrite with computed values
        return_dict[key] = metric_values[key].item()

    return return_dict


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, args=None, task=None):
    data_set = args.data_set
    # for bigearthnet, we use BCE loss
    if task.label_type == "multi_label_classification":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif (
        task.label_type == "segmentation"
    ):
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=0)

    metric_logger = helpers.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    metric = eval_metrics_generator(task).to(device) # obtain the eval metric

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.use_imnet_weights:
            # since we are making use of the weights trained on imagenet, we need to ensure the geobench is rgb. Hence if the data is bgr, we reaarange the channels
            if args.geobench_bands_type == "bgr":
                images = images[:, [2, 1, 0], :, :]


        # compute output
        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            output = model(images)
            if isinstance(output, dict):
                output = output["logits"]

            if task.label_type == "segmentation":
                output = output.permute(0, 2, 3, 1) # N, H, W, C

                output_tmp = output.contiguous().view(-1, output.size(3))
                target_tmp = target.unsqueeze(3)
                target_tmp = target_tmp.contiguous().view(-1, target_tmp.size(3))
                target_tmp = target_tmp.squeeze(1)
                target_tmp = target_tmp.long()
            else:
                if task.label_type == "multi_label_classification":
                    target_tmp = target.float()
                    output_tmp = output
                else:
                    target_tmp = target
                    output_tmp = output

            loss = criterion(output_tmp, target_tmp)

        if device.__str__ == "cuda":
            torch.cuda.synchronize()

    
        if (
            task.label_type == "segmentation"
        ):
            output = output.permute(0, 3, 1, 2)
            output = torch.nn.functional.softmax(output, dim=1) # argmax already applied in the metric
            target = target.squeeze(1)

        score = metric(output, target) # for bigearthnet, sigmoid is already applied in the metric

        batch_size = images.shape[0] # this can be used on the metric_logger update function if needed.
        metric_logger.update(loss=loss.item())
        for key in score.keys():
            metric_logger.meters[key].update(score[key].item())

    test_metric = metric.compute()
    logging_text = "**** "
    for key in test_metric.keys():
        logging_text += f"{key} {test_metric[key].item():.3f} "

    logging_text += f"loss {metric_logger.loss.global_avg:.3f}"

    print(logging_text)

    # we can compute global average on all except the metric values  
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # we replace the metrics with metric.compute() values
    for key in test_metric.keys():
        return_dict[key] = test_metric[key].item()

    return return_dict



    
