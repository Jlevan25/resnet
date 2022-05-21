from typing import Type

from configs.block_config import BlockConfig, BasicBlockConfig, BottleneckBlockConfig


class ResnetConfig:
    """
    #Resnet Struct
    forward(x):
    --> out = stem(x)
    --> out = stages(out)
    --> return final(out)
    """

    def __init__(self, in_channels: int, out_features: int, blocks: list, block_cfg: Type[BlockConfig]):
        self.stage_scale = 2
        self.start_filters = 64
        self.dilation = 1

        self.blocks = blocks
        self.block_cfg = block_cfg(scale=self.stage_scale)

        self.in_channels = in_channels
        self.out_features = out_features

    def get_stem_block(self, out_channels):
        return [('Conv2d', dict(in_channels=self.in_channels, out_channels=out_channels,
                                kernel_size=7, stride=2, padding=3, bias=True)),
                ('MaxPool2d', dict(kernel_size=3, stride=2, padding=1))]

    def get_final_block(self):
        in_features = self.start_filters * self.stage_scale ** (len(self.blocks) + 1)
        return [('AdaptiveAvgPool2d', dict(output_size=(1, 1))),
                ('Flatten', dict()),
                ('Linear', dict(in_features=in_features, out_features=self.out_features))]


class Resnet50Config(ResnetConfig):
    def __init__(self, in_channels: int, out_features: int):
        super().__init__(in_channels=in_channels, out_features=out_features,
                         blocks=[3, 4, 6, 3], block_cfg=BottleneckBlockConfig)


class Resnet34Config(ResnetConfig):
    def __init__(self, in_channels: int, out_features: int):
        super().__init__(in_channels=in_channels, out_features=out_features,
                         blocks=[3, 4, 6, 3], block_cfg=BasicBlockConfig)
