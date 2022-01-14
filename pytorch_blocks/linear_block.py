import torch.nn as nn
from pytorch_blocks.activation import get_activation
from pytorch_blocks.normalization import get_normalization


class LinearBlock(nn.Sequential):
    def __init__(self, in_dim, out_dim, act, norm):
        super.__init__(
            nn.Linear(in_dim, out_dim),
            get_activation(act),
            get_normalization(norm),
        )


class LinearBlocks(nn.Sequential):
    def __init__(self, num_blocks, dims, act, norm):
        super.__init__()
        assert len(dims) == num_blocks + 1
        for i in range(num_blocks):
            self.add_module(
                f'linear_block_{i}',
                LinearBlock(dims[i], dims[i + 1], act, norm)
            )

