# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import numpy as np

import torch
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics import JaccardIndex

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate, visualize_segmentation
from timm.loss import LabelSmoothingCrossEntropy
from custom_loss import DiceLoss


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            # make output class the last dimension
            if args.data_set == 'geobench.m-cashew-plantation' or args.data_set == 'geobench.m-SA-crop-type':
                output = output.permute(0, 2, 3, 1)
                # assuming for segmentation we have the cross entropy loss, we need to convert the output to something of shape N, C
                output_tmp = output.contiguous().view(-1, output.size(3))
                target_tmp = targets.unsqueeze(3)
                target_tmp = target_tmp.contiguous().view(-1, target_tmp.size(3))
                target_tmp = target_tmp.squeeze(1)
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
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            # loss.requires_grad = True
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()
        if args.data_set == 'geobench.m-bigearthnet':
            # for bigearthnet we calculate the mean average precision since that is used a lot for multi-label classification
            # we use the sigmoid function to get the probabilities
            out_p = torch.sigmoid(output)
            metric = MultilabelAveragePrecision(num_labels=43, average = 'macro')
            meanAP = metric(out_p, targets)
        elif args.data_set == 'geobench.m-cashew-plantation' or args.data_set == 'geobench.m-SA-crop-type':
            # for segmentation we calculate the mean intersection over union, hence the jaccard index
            output = output.permute(0, 3, 1, 2)
            out_p = torch.nn.functional.softmax(output, dim=1)
            out_p = torch.argmax(out_p, dim=1)
            # exit()
            num_classes = 7 if args.data_set == 'geobench.m-cashew-plantation' else 10
            # tmp = targets.unsqueeze(1)
            meanIoU = JaccardIndex(task = 'multiclass', num_classes=num_classes, average='macro').to(device)(out_p, targets)

        else:
            if mixup_fn is None:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
            else:
                class_acc = None


        metric_logger.update(loss=loss_value)
        if args.data_set == 'geobench.m-bigearthnet':
            metric_logger.update(meanAP=meanAP)
        elif args.data_set == 'geobench.m-cashew-plantation' or args.data_set == 'geobench.m-SA-crop-type':
            metric_logger.update(meanIoU=meanIoU)
        else:
            metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
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
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if args.data_set == 'geobench.m-bigearthnet':
                log_writer.update(meanAP=meanAP, head="loss")
            elif args.data_set == 'geobench.m-cashew-plantation' or args.data_set == 'geobench.m-SA-crop-type':
                log_writer.update(meanIoU=meanIoU, head="loss")
            else:
                log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, args=None):
    data_set = args.data_set
    # for bigearthnet, we use BCE loss
    if data_set == 'geobench.m-bigearthnet':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif data_set == 'geobench.m-cashew-plantation' or data_set == 'geobench.m-SA-crop-type':
        # criterion = DiceLoss(num_classes=args.nb_classes)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=0)
        # criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']

            if data_set == 'geobench.m-cashew-plantation' or data_set == 'geobench.m-SA-crop-type':
                output = output.permute(0, 2, 3, 1)

                output_tmp = output.contiguous().view(-1, output.size(3))
                target_tmp = target.unsqueeze(3)
                target_tmp = target_tmp.contiguous().view(-1, target_tmp.size(3))
                target_tmp = target_tmp.squeeze(1)
                target_tmp = target_tmp.long()
            else:
                if data_set == 'geobench.m-bigearthnet':
                    target_tmp = target.float()
                    output_tmp = output
                else:
                    target_tmp = target
                    output_tmp = output
            # loss = criterion(output, target)
            loss = criterion(output_tmp, target_tmp)

        torch.cuda.synchronize()
 
        
        # for bigearthnet we compute the mean average precision
        # we use the sigmoid function to get the probabilities
        if data_set == 'geobench.m-bigearthnet':
            out_p = torch.sigmoid(output)
            metric = MultilabelAveragePrecision(num_labels=43, average = 'macro')
            meanAP = metric(out_p, target)
        elif data_set == 'geobench.m-cashew-plantation' or data_set == 'geobench.m-SA-crop-type':
            # for segmentation we calculate the mean intersection over union, hence the jaccard index
            output = output.permute(0, 3, 1, 2)
            out_p = torch.nn.functional.softmax(output, dim=1)
            num_classes = 7 if data_set == 'geobench.m-cashew-plantation' else 10
            meanIoU = JaccardIndex(task = 'multiclass', num_classes=num_classes, average='macro').to(device)(out_p, target)
        else:
            # we use top 5 only if we have more than 5 classes
            if args.nb_classes > 5:
                acc = accuracy(output, target, topk=(1, 5))
            else:
                acc = accuracy(output, target, topk=(1,))

            if len(acc) == 2:
                acc1, acc5 = acc
            else:
                acc1 = acc[0]
                acc5 = None

            
        # # visualize some images and predictions
        # if args.visualize:
        #     if data_set == 'geobench.m-cashew-plantation' or data_set == 'geobench.m-SA-crop-type':
        #         output = output.permute(0, 2, 3, 1)
        #         out_p = torch.nn.functional.softmax(output, dim=3)
        #         out_p = torch.argmax(out_p, dim=3)
        #         visualize_segmentation(images, out_p, target, args, epoch=200)
        #         exit()
        #     else:
        #         raise NotImplementedError()


        ## for bigearthnet ONLY
        if data_set == 'geobench.m-bigearthnet':
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['meanAP'].update(meanAP.item(), n=batch_size)
        elif data_set == 'geobench.m-cashew-plantation' or data_set == 'geobench.m-SA-crop-type':
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['meanIoU'].update(meanIoU.item(), n=batch_size)
        else:
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            if acc5 is not None:
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    if data_set == 'geobench.m-bigearthnet':
        metric_logger.synchronize_between_processes()
        print("* Mean AP {meanAP.global_avg:.3f} loss {losses.global_avg:.3f}"
            .format(meanAP=metric_logger.meanAP, losses=metric_logger.loss))
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    elif data_set == 'geobench.m-cashew-plantation' or data_set == 'geobench.m-SA-crop-type':
        metric_logger.synchronize_between_processes()
        print("* Mean IoU {meanIoU.global_avg:.3f} loss {losses.global_avg:.3f}"
            .format(meanIoU=metric_logger.meanIoU, losses=metric_logger.loss))
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    else:
    # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if acc5 is not None:
            print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        else:
            print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
