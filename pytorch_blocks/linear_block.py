import torch.nn as nn
from pytorch_blocks.activation import get_activation
from pytorch_blocks.normalization import get_normalization


class LinearBlock(nn.Sequential):
    def __init__(self, act='none', norm='none', **linear_kwargs):
        super().__init__(
            nn.Linear(**linear_kwargs),
            get_activation(act),
            get_normalization(norm, num_features=linear_kwargs['out_features']),
        )


class LinearBlocks(nn.Sequential):
    def __init__(self, num_blocks, dims, act='none', norm='none', **linear_kwargs):
        super().__init__()
        assert len(dims) == num_blocks + 1
        for i in range(num_blocks):
            self.add_module(
                f'linear_block_{i}',
                LinearBlock(act, norm, in_features=dims[i], out_features=dims[i + 1], **linear_kwargs)
            )

