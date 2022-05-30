from utils import sum_except_dim, check_zero_divide

from metrics.metric import _ByClassMetric
from torch.nn import functional as F


class BalancedAccuracy(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        predictions = F.one_hot(predictions, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        if predictions.shape != targets.shape:
            targets = F.one_hot(targets, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        else:
            targets = targets.round()

        self._corrects += sum_except_dim(predictions * targets, dim=1).type(self._corrects.dtype)
        self._totals += sum_except_dim(targets, dim=1).type(self._totals.dtype)

        return check_zero_divide(self._corrects, self._totals)