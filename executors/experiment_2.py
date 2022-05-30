import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from executors.epoch_manager import EpochManager
from configs import Config, Resnet50Config
from models import Resnet
from metrics import BalancedAccuracy
from datasets import MixUpDatasetDecorator
from transforms import LabelSmoothing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'datasets')

cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
             dataset_name='AlmostCifar', out_features=12,
             model_name='Resnet50_tricks', device='cpu',
             batch_size=1, lr=5e-4, weight_decay=5e-4, momentum=0.9,
             debug=True, show_each=100,
             overfit=True, seed=None)

# model
model_cfg = Resnet50Config(in_channels=3, out_features=cfg.out_features)
model = Resnet(model_cfg).to(cfg.device)

keys = train_key, valid_key = 'train', 'valid'

if cfg.seed is not None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

jitter_param = (0.6, 1.4)
norm = [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.405],
                             std=[0.229, 0.224, 0.225])]

image_transforms = {train_key: transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=jitter_param,
                                                                          saturation=jitter_param,
                                                                          hue=(-.25, .25)),
                                                   *norm]),

                    valid_key: transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   *norm])}

target_transforms = {train_key: transforms.Compose([LabelSmoothing(cfg.out_features, alpha=0.1),
                                                    ])}


datasets_dict = {k: datasets.ImageFolder(root=os.path.join(DATASET_ROOT, train_key),
                                         transform=image_transforms[k] if k in image_transforms else None,
                                         target_transform=target_transforms[k] if k in target_transforms else None)
                 for k in keys}

if cfg.overfit:
    shuffle = False
    for k, dataset in datasets_dict.items():
        dataset.__len__ = lambda: cfg.batch_size
        datasets_dict[k] = dataset
else:
    shuffle = True

# Add Mixup
mixup_decorator = MixUpDatasetDecorator(cfg.out_features)
datasets_dict[train_key] = mixup_decorator(datasets_dict[train_key])

dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key],
                                          batch_size=cfg.batch_size, shuffle=shuffle),
                    valid_key: DataLoader(datasets_dict[valid_key],
                                          batch_size=cfg.batch_size, shuffle=shuffle)}

# weight decay
if cfg.weight_decay is not None:
    wd_params, no_wd_params = model.split_params4weight_decay()
    params = [dict(params=wd_params, weight_decay=cfg.weight_decay),
              dict(params=no_wd_params)]
else:
    params = model.parameters()

optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum)
criterion = nn.CrossEntropyLoss()
# scheduler = CosineAnnealingWarmRestarts(optimizer)
scheduler = CosineAnnealingLR(optimizer, 20)
writer = SummaryWriter(log_dir=cfg.LOG_PATH)

metrics = [BalancedAccuracy(model_cfg.out_features),
           ]
metrics_dict = {train_key: metrics, valid_key: metrics}

class_names = datasets_dict[train_key].classes
epoch_manager = EpochManager(dataloaders_dict=dataloaders_dict, class_names=class_names,
                             model=model, optimizer=optimizer, criterion=criterion, cfg=cfg,
                             scheduler=scheduler, writer=writer, metrics=metrics_dict)

epochs = 20

for epoch in range(epochs):
    # epoch_manager.train(train_key, epoch)
    # epoch_manager.save_model(epoch)

    for i, param_group in enumerate(epoch_manager.optimizer.param_groups):
        epoch_manager.writer.add_scalar(f'scheduler lr/param_group{i}',
                                        param_group['lr'], epoch)
    epoch_manager.validation(valid_key, epoch)
