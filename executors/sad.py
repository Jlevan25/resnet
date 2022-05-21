import os
import sys
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from executors import EpochManager, PhaseKeysDict
from configs import Config, Resnet50Config, BottleneckBlockConfig
from models import Resnet
from metrics import BalancedAccuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_cfg = Resnet50Config(in_channels=3, out_features=12)
cfg = Config(ROOT_DIR=ROOT, dataset_name='AlmostCifar', model_name='Resnet50',
             batch_size=128, lr=0.01, debug=True, show_each=100, device='cuda')

train_key, valid_key, test_key = 'train', 'valid', 'test'

jitter_param = (0.6, 1.4)
norm = [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.405],
                             std=[0.229, 0.224, 0.225])]

transforms = {train_key: transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=jitter_param,
                                                                    saturation=jitter_param,
                                                                    hue=(-.25, .25)),
                                             *norm]),

              valid_key: transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             *norm])}

datasets_dict = {train_key: datasets.ImageFolder(root=os.path.join(ROOT, 'datasets', train_key),
                                                 transform=transforms[train_key]),
                 valid_key: datasets.ImageFolder(root=os.path.join(ROOT, 'datasets', valid_key),
                                                 transform=transforms[valid_key])}

dataloaders = {train_key: DataLoader(datasets_dict[train_key],
                                     batch_size=cfg.batch_size, shuffle=True),
               valid_key: DataLoader(datasets_dict[valid_key],
                                     batch_size=cfg.batch_size)}

model = Resnet(model_cfg)
model.cuda()
#weight decay
wd_params, no_wd_params = model.split_params4weight_decay()
params = [dict(params=wd_params, weight_decay=1e-4),
          dict(params=no_wd_params)]

optimizer = optim.Adam(params, lr=cfg.lr)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer)
writer = SummaryWriter(log_dir=cfg.LOG_PATH)
metrics = [BalancedAccuracy(model_cfg.out_features)]

epoch_manager = EpochManager(model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             writer=writer,
                             dataloaders=dataloaders,
                             cfg=cfg,
                             scheduler=scheduler,
                             metrics=metrics,
                             phase_keys=PhaseKeysDict(train_key, valid_key, test_key))

epochs = 500

for epoch in range(epochs):
    epoch_manager.train(epoch)
    epoch_manager.save_model(epoch)

    epoch_manager.writer.add_scalar(f'scheduler lr',
                                    epoch_manager.optimizer.param_groups[0]['lr'], epoch)
    epoch_manager.validation(epoch)
