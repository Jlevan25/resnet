from torch import nn


def split_params4weight_decay(model):
    wd_params, no_wd = [], []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            wd_params.append(param)
        else:
            no_wd.append(param)
    return wd_params, no_wd


def zero_gamma_resnet(model):
    for name, module in model.__dict__['_modules'].items():
        if 'stage_' in name:
            # module_blocks = [list(m.block_layers)[::-1] for m in module]
            # last_batch_norm = [[layer for layer in m if 'BatchNorm' in type(layer).__name__][0] for m in module_blocks]
            last_batch_norm = [[layer for layer in blocks.block_layers if 'BatchNorm' in type(layer).__name__][-1]
                               for blocks in module]

            for batch_norm in last_batch_norm:
                nn.init.zeros_(batch_norm.weight)
