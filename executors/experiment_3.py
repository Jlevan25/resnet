import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from executors.epoch_manager import EpochManager
from configs import Config, Resnet50Config
from models import Resnet
from metrics import BalancedAccuracy
from datasets import MixUpDecorator, OverfitModeDecorator
from samplers import BatchUnderSampler
from transforms import LabelSmoothing, GaussianNoise
from utils import split_params4weight_decay, zero_gamma_resnet, LinearStochasticDepth, init_params, normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'datasets', 'data')

cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
             dataset_name='AlmostCifar', out_features=12,
             model_name='Resnet50_improvements', device='cpu',
             batch_size=1, lr=5e-4, weight_decay=5e-4, momentum=0.9,
             debug=True, show_each=100,
             overfit=True, seed=None)

# model
model_cfg = Resnet50Config(in_channels=3, out_features=cfg.out_features)
model = Resnet(model_cfg, stochastic_depth=LinearStochasticDepth).to(cfg.device)

keys = train_key, valid_key = 'filtered_train', 'valid'

if cfg.seed is not None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

jitter_param = (0.6, 1.4)
norm = transforms.Normalize(mean=[0.5764, 0.5342, 0.4983], std=[0.2365, 0.2404, 0.2410])

image_transforms = {train_key: transforms.Compose([transforms.RandomResizedCrop(352),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=jitter_param,
                                                                          saturation=jitter_param,
                                                                          hue=(-.25, .25)),
                                                   transforms.RandomAffine(degrees=(-15, 15),
                                                                           translate=(0.0, 0.15),
                                                                           scale=(0.9, 1.)),
                                                   transforms.ToTensor(),
                                                   GaussianNoise(mean=[0.5764, 0.5342, 0.4983],
                                                                 std=[0.2365, 0.2404, 0.2410]),
                                                   norm]),

                    valid_key: transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   norm])}

target_transforms = {train_key: transforms.Compose([LabelSmoothing(cfg.out_features, alpha=0.1),
                                                    ])}

datasets_dict = {k: datasets.ImageFolder(root=os.path.join(DATASET_ROOT, k),
                                         transform=image_transforms[k] if k in image_transforms else None,
                                         target_transform=target_transforms[k] if k in target_transforms else None)
                 for k in keys}

if cfg.overfit:
    # overfit
    shuffle = False
    overfit_mode = OverfitModeDecorator(cfg.batch_size)
    for key in datasets_dict.keys():
        datasets_dict[key] = overfit_mode(datasets_dict[key])
    train_kwargs = dict(batch_size=cfg.batch_size)
else:
    # under sample
    shuffle = True
    classes_count = torch.from_numpy(np.bincount(datasets_dict[train_key].targets))
    indexes = {c: [] for c, _ in enumerate(classes_count)}
    for i, t in enumerate(datasets_dict[train_key].targets):
        indexes[t].append(i)
    batch_sampler = BatchUnderSampler(indexes=list(indexes.values()), threshold=classes_count.min(),
                                      batch_size=cfg.batch_size, shuffle=shuffle)
    train_kwargs = dict(batch_sampler=batch_sampler)

# Add Mixup
mixup_decorator = MixUpDecorator(cfg.out_features)
datasets_dict[train_key] = mixup_decorator(datasets_dict[train_key])

dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key], **train_kwargs),
                    valid_key: DataLoader(datasets_dict[valid_key], batch_size=cfg.batch_size)}

# linear he_init
init_params(model, 'Linear', weight_init_def=nn.init.kaiming_uniform_)

# zero gamma in batch norm
zero_gamma_resnet(model)

# weight decay
if cfg.weight_decay is not None:
    wd_params, no_wd_params = split_params4weight_decay(model)
    params = [dict(params=wd_params, weight_decay=cfg.weight_decay),
              dict(params=no_wd_params)]
else:
    params = model.parameters()

optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(log_dir=cfg.LOG_PATH)

metrics = [BalancedAccuracy(model_cfg.out_features),
           ]
metrics_dict = {train_key: metrics, valid_key: metrics}

epochs = 20
end_warmup = 4
scheduler = [LinearLR(optimizer, start_factor=0.1, total_iters=end_warmup),
             CosineAnnealingLR(optimizer, epochs - end_warmup)]

class_names = datasets_dict[train_key].classes
epoch_manager = EpochManager(dataloaders_dict=dataloaders_dict, class_names=class_names,
                             model=model, optimizer=optimizer, criterion=criterion, cfg=cfg,
                             scheduler=scheduler, writer=writer, metrics=metrics_dict)

save_each = 5
for epoch in range(epochs):
    if epoch == end_warmup:
        epoch_manager.switch_scheduler()

    epoch_manager.train(train_key, epoch)
    if epoch % save_each == 0 and epoch != 0:
        epoch_manager.save_model(epoch)

    for i, param_group in enumerate(epoch_manager.optimizer.param_groups):
        epoch_manager.writer.add_scalar(f'scheduler lr/param_group{i}',
                                        param_group['lr'], epoch)
    epoch_manager.validation(valid_key, epoch)


