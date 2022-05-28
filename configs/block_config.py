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

    def __init__(self, scale=2, bias=False):
        self.scale = scale
        self.bias = bias

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

    def conv3x3(self, in_channels, out_channels, stride=1, groups=1, dilation=1, bias=None):
        """3x3 convolution with padding"""
        return 'Conv2d', dict(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=stride, padding=dilation,
                              groups=groups, dilation=dilation,
                              bias=self.bias if bias is None else bias)

    def conv1x1(self, in_channels, out_channels, stride=1, bias=None):
        """1x1 convolution"""
        return 'Conv2d', dict(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1, stride=stride, bias=self.bias if bias is None else bias)


class BasicBlock(BlockConfig):

    def get_block(self, in_channels, stage_start: bool = False, first_stage: bool = False):
        stage_start = False if first_stage else stage_start
        return [self.conv3x3(in_channels, in_channels, stride=(2 if stage_start else 1)),
                ('BatchNorm2d', dict(num_features=in_channels)),
                ('ReLU', dict(inplace=True)),
                self.conv3x3(in_channels, in_channels),
                ('BatchNorm2d', dict(num_features=in_channels))]

    def get_block_residual(self):
        return [('ReLU', dict(inplace=True))]

