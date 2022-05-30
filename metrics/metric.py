from abc import abstractmethod, ABC

import torch
from utils import check_negative_divide


class Metric:

    @abstractmethod
    def get_epoch_metric(self):
        raise NotImplementedError()

    @abstractmethod
    def get_batch_metric(self, predictions, targets):
        raise NotImplementedError()


class _ByClassMetric(Metric, ABC):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes
        self._corrects = -torch.ones(self.classes)
        self._totals = -torch.ones(self.classes)

    def get_epoch_metric(self):
        mean = check_negative_divide(self._corrects, self._totals)
        self._corrects = -torch.ones(self.classes)
        self._totals = -torch.ones(self.classes)
        return mean

    def _check_negative(self, index):
        if any(self._totals[index] < 0):
            self._corrects[index] = 0
            self._totals[index] = 0
