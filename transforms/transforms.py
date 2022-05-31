import cv2
import numpy as np
import torch
from torch import nn

from utils import normalize


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, image):
        if len(image.shape) > 2:
            for c in range(len(image)):
                image[c] += torch.from_numpy(np.random.normal(self.mean[c], self.std[c], image.shape[1:]))
        else:
            image += torch.from_numpy(np.random.normal(self.mean, self.std, image.shape))

        # img = cv2.cvtColor(np.asarray(image).transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
        # img_norm = np.uint8(normalize(img) * 255)
        # cv2.imshow('img', img)
        # cv2.waitKey()

        return image

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'