from utils import sum_except_dim, check_zero_divide, check_negative_divide

from metrics.metric import _ByClassMetric
from torch.nn import functional as F


class BalancedAccuracy(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        predictions = F.one_hot(predictions, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        if predictions.shape != targets.shape:
            targets = F.one_hot(targets, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        else:
            targets = targets.round()

        correct = sum_except_dim(predictions * targets, dim=1).type(self._corrects.dtype)
        total = sum_except_dim(targets, dim=1).type(self._totals.dtype)

        self._check_negative(total > 0)

        self._corrects += correct
        self._totals += total

        return check_negative_divide(self._corrects, self._totals)
