import torch.nn as nn
from pytorch_blocks.activation import get_activation
from pytorch_blocks.normalization import get_normalization


class ConvBlock(nn.Sequential):
    def __init__(self, act='none', norm='none', **conv_kargs):
        super().__init__(
            nn.Conv2d(**conv_kargs),
            get_activation(act),
            get_normalization(norm, num_features=conv_kargs['out_channels']),
        )


class ConvBlocks(nn.Sequential):
    def __init__(self, num_blocks, dims, act='none', norm='none', **conv_kargs):
        super().__init__()
        assert len(dims) == num_blocks + 1
        for i in range(num_blocks):
            self.add_module(
                f'conv_block_{i}',
                ConvBlock(act, norm, in_channels=dims[i], out_channels=dims[i + 1], **conv_kargs)
            )


