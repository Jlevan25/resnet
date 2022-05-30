import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from executors.epoch_manager import EpochManager
from configs import Config, Resnet50Config
from models import Resnet
from metrics import BalancedAccuracy
from datasets import OverfitModeDecorator
from utils import split_params4weight_decay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, 'datasets')

cfg = Config(ROOT_DIR=ROOT, DATASET_DIR=DATASET_ROOT,
             dataset_name='AlmostCifar', out_features=12,
             model_name='Resnet50_baseline', device='cuda',
             batch_size=128, lr=5e-4, weight_decay=5e-4, momentum=0.9,
             debug=True, show_each=100,
             overfit=False, seed=None)

model_cfg = Resnet50Config(in_channels=3, out_features=cfg.out_features)

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
target_transforms = {}

datasets_dict = {k: datasets.ImageFolder(root=os.path.join(DATASET_ROOT, k),
                                         transform=image_transforms[k] if k in image_transforms else None,
                                         target_transform=target_transforms[k] if k in target_transforms else None)
                 for k in keys}

if cfg.overfit:
    shuffle = False
    overfit_mode = OverfitModeDecorator(cfg.batch_size)
    for key in datasets_dict.keys():
        datasets_dict[key] = overfit_mode(datasets_dict[key])
else:
    shuffle = True

dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key],
                                          batch_size=cfg.batch_size, shuffle=shuffle),
                    valid_key: DataLoader(datasets_dict[valid_key],
                                          batch_size=cfg.batch_size)}

model = Resnet(model_cfg).to(cfg.device)

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

metrics = [BalancedAccuracy(model_cfg.out_features)]
metrics_dict = {train_key: metrics, valid_key: metrics}

class_names = datasets_dict[train_key].classes
epoch_manager = EpochManager(dataloaders_dict=dataloaders_dict, class_names=class_names,
                             model=model, optimizer=optimizer, criterion=criterion, cfg=cfg,
                             writer=writer, metrics=metrics_dict)

epochs = 20
save_each = 5

for epoch in range(epochs):
    epoch_manager.train(train_key, epoch)
    if epoch % save_each == 0 and epoch != 0:
        epoch_manager.save_model(epoch)

    for i, param_group in enumerate(epoch_manager.optimizer.param_groups):
        epoch_manager.writer.add_scalar(f'scheduler lr/param_group{i}',
                                        param_group['lr'], epoch)
    epoch_manager.validation(valid_key, epoch)
