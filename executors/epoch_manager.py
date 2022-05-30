import os.path
from typing import Iterator, Union, List

import torch
from torch import tensor
from torch.utils.data import DataLoader


# from datasets import CocoLocalizationDataset


class EpochManager:
    def __init__(self,
                 model, optimizer, criterion, cfg,
                 writer=None,
                 scheduler: Union[List, ] = None,
                 class_names=None,
                 dataloaders_dict=None,
                 metrics: dict = None):
        self.cfg = cfg
        self.model = model

        self.optimizer = optimizer
        if hasattr(scheduler, '__len__'):
            self._scheduler_list = scheduler
            self._scheduler_index = 0
            self.scheduler = scheduler[0]
        else:
            self._scheduler_list = None
            self.scheduler = scheduler

        self.criterion = criterion
        self.metrics = metrics
        self.device = self.cfg.device
        self.writer = writer

        self.class_names = class_names if class_names is not None else list(range(cfg.out_features))
        self.dataloaders = dataloaders_dict if dataloaders_dict is not None else dict()

        self._global_step = dict()

    def switch_scheduler(self, index=None):
        if index is None:
            self._scheduler_index += 1 if self._scheduler_index <= len(self._scheduler_list) else 0
        else:
            self._scheduler_index = index
        self.scheduler = self._scheduler_list[self._scheduler_index]

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1

    @torch.no_grad()
    def _calc_epoch_metrics(self, stage):
        for metric in self.metrics[stage]:
            self._write_metrics(metric_values=metric.get_epoch_metric(),
                                metric_name=type(metric).__name__,
                                stage=stage, debug=self.cfg.debug)

    @torch.no_grad()
    def _calc_batch_metrics(self, predictions, targets, stage, debug):
        for metric in self.metrics[stage]:
            self._write_metrics(metric_values=metric.get_batch_metric(predictions, targets),
                                metric_name=type(metric).__name__,
                                stage=stage, debug=debug)

    def _write_scalar(self, tag, value, step):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _write_metrics(self, metric_values, metric_name, stage, debug):

        if len(metric_values) > 1:
            if self.cfg.write_by_class_metrics:
                for cls, scalar in zip(self.class_names, metric_values):
                    if scalar >= 0:
                        self._write_scalar(f'{stage}/{metric_name}/{cls}', scalar.item(), self._global_step[stage])

            mean_value = metric_values[metric_values >= 0]
            mean_value = mean_value.mean().item() if len(mean_value) > 0 else 0.
            self._write_scalar(f'{stage}/Mean/{metric_name}', mean_value, self._global_step[stage])

            if debug:
                metric_values[metric_values < 0] = 0.
                if self.cfg.write_by_class_metrics:
                    print("{}: {}".format(metric_name, metric_values.tolist()))

                print("Mean {}: {}".format(metric_name, mean_value))

        else:
            metric_values = metric_values.item()
            self._write_scalar(f'{stage}/{metric_name}', metric_values, self._global_step[stage])

            if debug:
                print("{}: {}".format(metric_name, metric_values))

    def _epoch_generator(self, stage, epoch=None) -> Iterator[tensor]:

        # if stage not in self.dataloaders:
        #     self._get_data(stage)

        if stage not in self._global_step:
            self._get_global_step(stage)

        calc_metrics = self.metrics[stage] is not None and len(self.metrics[stage]) > 0
        print('\n_______', stage, f'epoch{epoch}' if epoch is not None else '',
              'len:', len(self.dataloaders[stage]), '_______')

        for i, (images, targets) in enumerate(self.dataloaders[stage]):

            self._global_step[stage] += 1
            debug = self.cfg.debug and i % self.cfg.show_each == 0

            predictions = self.model(images.to(self.device))

            loss = self.criterion(predictions, targets.to(self.device))
            self._write_scalar(f'{stage}/loss', loss, self._global_step[stage])

            if debug:
                print('\n___', f'Iteration {i}', '___')
                print(f'Loss: {loss.item()}')

            if calc_metrics:
                self._calc_batch_metrics(predictions.argmax(1).cpu(), targets.cpu(), stage, debug)

            yield loss

        if calc_metrics:
            print('\n___', f'Epoch Summary', '___')
            self._calc_epoch_metrics(stage)

    def train(self, stage_key, i_epoch):
        self.model.train()
        for batch_loss in self._epoch_generator(stage=stage_key, epoch=i_epoch):
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def validation(self, stage_key, i_epoch):
        self.model.eval()
        for batch_loss in self._epoch_generator(stage=stage_key, epoch=i_epoch):
            ...

    @torch.no_grad()
    def test(self, stage_key):
        self.model.eval()
        for batch_loss in self._epoch_generator(stage=stage_key):
            ...

    def save_model(self, epoch, path=None):
        path = self.cfg.SAVE_PATH if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, f'{epoch}.pth')

        checkpoint = dict(epoch=self._global_step,
                          model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict())

        torch.save(checkpoint, path)
        print('model saved, epoch:', epoch)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self._global_step = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('model loaded')
