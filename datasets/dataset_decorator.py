from abc import abstractmethod

import numpy as np


class DatasetDecorator:
    def __call__(self, *args, **kwargs):
        return self.decorate(*args, **kwargs)

    @abstractmethod
    def decorate(self, *args, **kwargs):
        raise NotImplementedError()


class _OvefitedDataset:
    def __init__(self, batch_size, dataset):
        self._batch_size = batch_size
        self._dataset = dataset
        self._dataset.__len__ = lambda: batch_size

    def __getitem__(self, item):
        return self._dataset[item]

    def __len__(self):
        return self._batch_size

    def __repr__(self):
        return repr(self._dataset)


class OverfitModeDecorator(DatasetDecorator):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def decorate(self, dataset):
        _OvefitedDataset.__name__ = 'Ovefit' + type(dataset).__name__
        overfit_dataset = _OvefitedDataset(self.batch_size, dataset)
        for k, v in dataset.__dict__.items():
            overfit_dataset.__dict__[k] = v
        return overfit_dataset


class MixUpDecorator(DatasetDecorator):

    def __init__(self, num_classes: int, alpha: float = 1.0):
        self.num_classes = num_classes
        self.alpha = alpha if alpha > 0 else 0

    def decorate(self, dataset):
        func = dataset.__getitem__

        def wrapper(index: int):
            img1, target1 = func(index)
            second_idx = np.random.uniform(0, self.num_classes)
            if second_idx == index:
                return img1, target1

            lmd = round(np.random.beta(self.alpha, self.alpha), 4)
            if lmd == 0.5:
                lmd = 0.5001

            img2, target2 = func(second_idx)
            mix_img = lmd * img1 + (1 - lmd) * img2
            mix_target = lmd * target1 + (1 - lmd) * target2 if target1 != target2 else target1

            return mix_img, mix_target

        dataset.__getitem__ = wrapper

        return dataset
