import time
import torch


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} in {end - start:0.8f} seconds")
        return result

    return wrapper


def one_hot_argmax(tensor):
    batch_size, classes, *dims = tensor.shape
    preds = torch.zeros(tensor.nelement() // classes, classes)
    preds[torch.arange(len(preds)), tensor.argmax(dim=1).reshape(-1)] = 1
    return preds.reshape(batch_size, *dims, classes).permute(0, -1, *(torch.arange(len(dims)) + 1))


def sum_except_dim(x, dim):
    return x.transpose(dim, 0).reshape(x.shape[dim], -1).sum(1)


def check_negative_divide(x, y):
    if any(y < 0):
        out = -torch.ones_like(x)
        indexes = y > 0
        out[indexes] = x[indexes] / y[indexes]
    else:
        out = x / y

    return out


def check_zero_divide(x, y):
    if any(y == 0):
        out = torch.zeros_like(x)
        indexes = y > 0
        out[indexes] = x[indexes] / y[indexes]
    else:
        out = x / y

    return out


def calc_mean_std(stages, dataloaders, show_each):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_data = 0
    stages = stages if hasattr(stages, '__len__') else [stages]

    for stage in stages:
        print(stage, len(dataloaders[stage]))
        for i, (images, _) in enumerate(dataloaders[stage]):
            mean += images.mean((0, 2, 3))
            std += images.std((2, 3)).mean(0)
            num_data += 1
            if i % show_each == 0:
                print(i, 'mean:', mean/num_data, 'std:', std/num_data)

    print('overall', num_data)
    print('mean:', mean/num_data, 'std:', std/num_data)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())