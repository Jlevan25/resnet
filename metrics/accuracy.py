from utils import sum_except_dim, check_zero_divide

from metrics.metric import _ByClassMetric


class BalancedAccuracy(_ByClassMetric):

    def get_batch_metric(self, predictions, targets):
        predictions = F.one_hot(predictions, num_classes=self.classes).transpose(1, -1).squeeze(-1)
        targets = F.one_hot(targets, num_classes=self.classes).transpose(1, -1).squeeze(-1)

        correct = sum_except_dim(predictions * targets, dim=1)
        total = sum_except_dim(targets, dim=1)
        self._corrects += correct.type(self._corrects.dtype)
        self._totals += total.type(self._totals.dtype)

        return check_zero_divide(correct, total)