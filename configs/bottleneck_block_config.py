from configs.block_config import BlockConfig


class BottleneckBlock(BlockConfig):
    def __init__(self, scale, bias=False):
        self.bottleneck_reduction = 4
        super().__init__(scale, bias)

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
            out_channels = in_channels * self.scale ** 2
        else:
            in_channels *= self.scale
            out_channels = in_channels * self.scale

        return [self.conv1x1(in_channels, out_channels, stride=(2 if not first_stage else 1)),
                ('BatchNorm2d', dict(num_features=out_channels))]


# TODO: bad name
class BottleneckBlockBD(BottleneckBlock):

    def get_block(self, in_channels, stage_start: bool = False, first_stage: bool = False):
        stride = 2 if stage_start and not first_stage else 1
        in_channels, bn_channels, out_channels = self._get_channels(in_channels, stage_start, first_stage)

        return [self.conv1x1(in_channels, bn_channels),
                ('BatchNorm2d', dict(num_features=bn_channels)),
                ('ReLU', dict(inplace=True)),
                self.conv3x3(bn_channels, bn_channels, stride=stride),
                ('BatchNorm2d', dict(num_features=bn_channels)),
                ('ReLU', dict(inplace=True)),
                self.conv1x1(bn_channels, out_channels),
                ('BatchNorm2d', dict(num_features=out_channels))]

    def get_stage_residual(self, in_channels, first_stage: bool):
        if first_stage:
            return super().get_stage_residual(in_channels, first_stage)

        in_channels *= self.scale
        out_channels = in_channels * self.scale

        return [('AvgPool2d', dict(kernel_size=(2, 2), stride=2)),
                self.conv1x1(in_channels, out_channels),
                ('BatchNorm2d', dict(num_features=out_channels))]
