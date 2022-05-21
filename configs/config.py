import os
import time


class Config(object):
    def __init__(self, batch_size, lr, dataset_name, model_name, ROOT_DIR=None, debug=True, show_each=1, device='cpu'):
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if ROOT_DIR is None else ROOT_DIR
        self.debug = debug
        self.batch_size = batch_size
        self.lr = lr
        self.show_each = show_each
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device

        experiment_name = f'model_{self.model_name}_batch_size{self.batch_size}_lr_{self.lr}_{time.time()}'

        self.SAVE_PATH = os.path.join(self.ROOT_DIR, 'checkpoints', self.model_name,
                                      self.dataset_name, str(time.time()))

        self.LOG_PATH = os.path.join(self.ROOT_DIR, 'logs', self.dataset_name, experiment_name)
