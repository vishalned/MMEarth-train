from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Dice


class UncertaintyWeightingStrategy(nn.Module):
    """Uncertainty weighting strategy"""

    def __init__(self, tasks):
        super(UncertaintyWeightingStrategy, self).__init__()

        self.tasks = tasks
        self.log_vars = nn.Parameter(torch.zeros(tasks))

    def forward(self, task_losses: List[Tensor]):
        losses_tensor = torch.stack(task_losses)
        non_zero_losses_mask = losses_tensor != 0.0

        # calculate weighted losses
        losses_tensor = torch.exp(-self.log_vars) * losses_tensor + self.log_vars

        # if some loss was 0 (i.e. task was dropped), weighted loss should also be 0 and not just log_var as no information was gained
        losses_tensor *= non_zero_losses_mask

        weighted_task_losses = losses_tensor
        return weighted_task_losses, self.log_vars.tolist()


class LabelSmoothingBinaryCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.0, reduction: str = "mean"):
        super(LabelSmoothingBinaryCrossEntropy, self).__init__()
        assert 0 <= smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = (
                target * positive_smoothed_labels
                + (1 - target) * negative_smoothed_labels
            )

        loss = self.bce_with_logits(input, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes=None):
        super(DiceLoss, self).__init__()
        self.dice = Dice(num_classes=num_classes, average="macro").to("cuda")

    def forward(self, input, target):
        input = F.softmax(input, dim=-1)
        input = input.argmax(dim=-1)  # This is non-differentiable
        input = input.squeeze()
        return 1 - self.dice(input, target)
