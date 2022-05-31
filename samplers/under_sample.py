from typing import Optional, Sized, Iterator, List

import numpy as np
from torch.utils.data import Sampler
from random import shuffle as shuffle_


class BatchUnderSampler(Sampler):
    def __init__(self, threshold: int, indexes, batch_size, shuffle: bool = True, drop_last: bool = False):
        super().__init__(None)
        self.threshold = threshold
        self.indexes = indexes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:

        if self.shuffle:
            for ids in self.indexes:
                shuffle_(ids)

        indexes = np.array([ids[:self.threshold] for ids in self.indexes]).T.flatten().tolist()

        batch = []
        for idx in indexes:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        l = (self.threshold * len(self.indexes))
        if self.drop_last:
            return l // self.batch_size  # type: ignore[arg-type]
        else:
            return (l + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
