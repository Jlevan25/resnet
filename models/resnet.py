from torch import nn
import torch
from configs.block_config import BlockConfig
from configs.resnet_config import ResnetConfig


def _make_layers(cfg: list):
    if cfg is None:
        return None

    layers = []
    for layer, kwargs in cfg:
        if layer in nn.modules.__all__:
            layers.append(eval(f'{nn.__name__}.{layer}')(**kwargs))
        else:
            raise ValueError(f'There is no {layer} among torch.nn modules')
    return nn.Sequential(*layers)


class _Block(nn.Module):
    def __init__(self, in_channels: int, cfg: BlockConfig, stage_start: bool, first_stage: bool):
        super().__init__()
        self.block_layers = _make_layers(cfg.get_block(in_channels, stage_start, first_stage))
        self.block_residual = _make_layers(cfg.get_block_residual())
        self.stage_residual = _make_layers(cfg.get_stage_residual(in_channels, first_stage)) if stage_start else None

    def forward(self, x):
        out = self.block_layers(x)
        out += self.stage_residual(x) if self.stage_residual is not None else x
        if self.block_residual is not None:
            out = self.block_residual(out)

        return out


class Resnet(nn.Module):
    def __init__(self, cfg: ResnetConfig):
        super().__init__()
        out_d = cfg.start_filters

        self.stem = _make_layers(cfg.get_stem_block(out_d))
        for stage, num_blocks in enumerate(cfg.blocks):
            setattr(self, f'stage_{stage + 1}', nn.Sequential(*[_Block(in_channels=out_d,
                                                                       stage_start=(i == 0),
                                                                       first_stage=(stage == 0),
                                                                       cfg=cfg.block_cfg)
                                                                for i in range(num_blocks)]))
            out_d *= cfg.stage_scale

        self.final_stage = _make_layers(cfg.get_final_block())
        # initialize the weights
        self.initialize_parameters()

    def initialize_parameters(self,
                              weights_init_def=nn.init.xavier_uniform_,
                              bias_init_def=nn.init.zeros_,
                              weights_kwargs=None,
                              bias_kwargs=None):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                weights_init_def(module.weight, **weights_kwargs) if weights_kwargs is not None \
                    else weights_init_def(module.weight)
                if module.bias is not None:
                    bias_init_def(module.bias, **bias_kwargs) if bias_kwargs is not None \
                        else weights_init_def(module.weight)

    def split_params4weight_decay(self):
        wd_params, no_wd = [], []
        for name, param in self.named_parameters():
            if name.split('.')[-1] == 'weight':
                wd_params.append(param)
            else:
                no_wd.append(param)
        return wd_params, no_wd

    def forward(self, x):
        out = self.stem(x)
        for name, module in self.__dict__['_modules'].items():
            if 'stage_' in name:
                out = module(out)

        return self.final_stage(out)


if __name__ == '__main__':
    from torchsummary import summary
    from configs import Resnet50Config

    cfg = Resnet50Config(3, 10)
    model = Resnet(cfg)
    model.split_params4weight_decay()

    summary = summary(model, (3, 224, 224), batch_size=32)
    print(model)
