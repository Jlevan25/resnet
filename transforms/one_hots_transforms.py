import torch
from torch import nn
from torch.nn import functional as F


class ToOneHot(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, target):
        return F.one_hot(target, num_classes=self.classes).transpose(1, -1).squeeze(-1)


class LabelSmoothing(nn.Module):
    def __init__(self, num_classes: int, alpha: float):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, target):
        if not hasattr(target, '__len__'):
            target = F.one_hot(torch.tensor(target), num_classes=self.num_classes)
        equal = (1. - self.alpha) * target
        otherwise = (target < 1) * self.alpha / (self.num_classes - 1)
        return equal + otherwise
