from abc import abstractmethod


class BlockConfig:
    """
    #Block Struct
    forward(x):
    --> out = block(x)
    --> x = stage_residual(x) if stage_start
    --> out += x
    --> block_residual(out)
    """

    def __init__(self, scale=2):
        self.scale = scale

    @abstractmethod
    def get_block_residual(self):
        raise NotImplementedError()

    def get_stage_residual(self, in_channels, first_stage: bool):
        out_channels = in_channels * self.scale
        return [self.conv1x1(in_channels, out_channels, stride=(2 if not first_stage else 1)),
                ('BatchNorm2d', dict(num_features=out_channels))]

    @abstractmethod
    def get_block(self, in_channels, stage_start: bool, first_stage: bool):
        raise NotImplementedError()

    @staticmethod
    def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=True):
        """3x3 convolution with padding"""
        return 'Conv2d', dict(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=stride, padding=dilation,
                              groups=groups, bias=bias, dilation=dilation)

    @staticmethod
    def conv1x1(in_channels, out_channels, stride=1, bias=True):
        """1x1 convolution"""
        return 'Conv2d', dict(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1, stride=stride, bias=bias)


class BasicBlockConfig(BlockConfig):

    def get_block(self, in_channels, stage_start: bool = False, first_stage: bool = False):
        stage_start = False if first_stage else stage_start
        return [self.conv3x3(in_channels, in_channels, stride=(2 if stage_start else 1)),
                ('BatchNorm2d', dict(num_features=in_channels)),
                ('ReLU', dict(inplace=True)),
                self.conv3x3(in_channels, in_channels),
                ('BatchNorm2d', dict(num_features=in_channels))]

    def get_block_residual(self):
        return [('ReLU', dict(inplace=True))]


class BottleneckBlockConfig(BlockConfig):
    def __init__(self, scale):
        self.bottleneck_reduction = 4
        super().__init__(scale)

    def _get_channels(self, in_channels, stage_start: bool, first_stage: bool):
        if first_stage and stage_start:
            scale = self.scale ** 2
        elif stage_start:
            scale = self.scale
            in_channels *= scale
        else:
            scale = 1
            in_channels *= self.scale ** 2

        out_channels = in_channels * scale
        bottleneck_channels = out_channels // self.bottleneck_reduction

        return in_channels, bottleneck_channels, out_channels

    def get_block(self, in_channels, stage_start: bool = False, first_stage: bool = False):
        stride = 2 if stage_start and not first_stage else 1
        in_channels, bn_channels, out_channels = self._get_channels(in_channels, stage_start, first_stage)

        return [self.conv1x1(in_channels, bn_channels, stride=stride),
                ('BatchNorm2d', dict(num_features=bn_channels)),
                ('ReLU', dict(inplace=True)),
                self.conv3x3(bn_channels, bn_channels),
                ('BatchNorm2d', dict(num_features=bn_channels)),
                ('ReLU', dict(inplace=True)),
                self.conv1x1(bn_channels, out_channels),
                ('BatchNorm2d', dict(num_features=out_channels))]

    def get_block_residual(self):
        return [('ReLU', dict(inplace=True))]

    def get_stage_residual(self, in_channels, first_stage: bool):
        if first_stage:
            out_channels = in_channels * self.scale**2
        else:
            in_channels *= self.scale
            out_channels = in_channels * self.scale

        return [self.conv1x1(in_channels, out_channels, stride=(2 if not first_stage else 1)),
                ('BatchNorm2d', dict(num_features=out_channels))]
