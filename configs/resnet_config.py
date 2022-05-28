from typing import Type

from configs.block_config import BlockConfig, BasicBlock
from configs.bottleneck_block_config import BottleneckBlock, BottleneckBlockBD


class ResnetConfig:
    """
    #Resnet Struct
    forward(x):
    --> out = stem(x)
    --> out = stages(out)
    --> return final(out)
    """

    def __init__(self, in_channels: int, out_features: int, blocks: list, block_cfg: Type[BlockConfig],
                 bias: bool = False):
        self.stage_scale = 2
        self.start_filters = 64
        self.dilation = 1
        self.bias = bias

        self.blocks = blocks
        self.block_cfg = block_cfg(scale=self.stage_scale, bias=bias)

        self.in_channels = in_channels
        self.out_features = out_features

    def get_stem_block(self, out_channels):
        return [('Conv2d', dict(in_channels=self.in_channels, out_channels=out_channels,
                                kernel_size=7, stride=2, padding=3, bias=self.bias)),
                ('MaxPool2d', dict(kernel_size=3, stride=2, padding=1))]

    def get_final_block(self):
        in_features = self.start_filters * self.stage_scale ** (len(self.blocks) + 1)
        return [('AdaptiveAvgPool2d', dict(output_size=(1, 1))),
                ('Flatten', dict()),
                ('Linear', dict(in_features=in_features, out_features=self.out_features))]


class Resnet50Config(ResnetConfig):
    def __init__(self, in_channels: int, out_features: int, bias: bool = False):
        super().__init__(in_channels=in_channels, out_features=out_features,
                         blocks=[3, 4, 6, 3], block_cfg=BottleneckBlock, bias=bias)


class Resnet50BCDConfig(ResnetConfig):
    def __init__(self, in_channels: int, out_features: int, bias: bool = False):
        super().__init__(in_channels=in_channels, out_features=out_features,
                         blocks=[3, 4, 6, 3], block_cfg=BottleneckBlockBD, bias=bias)

    def get_stem_block(self, out_channels):
        return [('Conv2d', dict(in_channels=self.in_channels, out_channels=out_channels,
                                kernel_size=3, stride=2, padding=1, bias=self.bias)),
                ('Conv2d', dict(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1, bias=self.bias)),
                ('Conv2d', dict(in_channels=out_channels, out_channels=out_channels,
                                kernel_size=3, stride=1, padding=1, bias=self.bias)),
                ('MaxPool2d', dict(kernel_size=3, stride=2, padding=1))]


class Resnet34Config(ResnetConfig):
    def __init__(self, in_channels: int, out_features: int, bias: bool = False):
        super().__init__(in_channels=in_channels, out_features=out_features,
                         blocks=[3, 4, 6, 3], block_cfg=BottleneckBlock, bias=bias)
