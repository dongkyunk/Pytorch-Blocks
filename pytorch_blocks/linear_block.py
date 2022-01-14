import torch.nn as nn
from pytorch_blocks.activation import get_activation
from pytorch_blocks.normalization import get_normalization


class LinearBlock(nn.Sequential):
    """Linear block with activation and normalization.

    Args:
        act (str): activation function name.
        norm (str): normalization function name.
        **conv_kargs: keyword arguments for nn.Linear.
    """    
    def __init__(self, act='none', norm='none', **linear_kwargs):
        assert norm in ['bn1d', 'in1d', 'ln', 'none'], "Unsupported normalization function."
        super().__init__(
            nn.Linear(**linear_kwargs),
            get_normalization(norm, num_features=linear_kwargs['out_features']),
            get_activation(act),
        )


class LinearBlocks(nn.Sequential):
    """Linear blocks with activation and normalization.

    Args:
        num_blocks (int): number of linear blocks.
        dims (list): list of input and output channels.
        act (str): activation function name.
        norm (str): normalization function name.
        **conv_kargs: keyword arguments for nn.Linear.
    """    
    def __init__(self, num_blocks, dims, act='none', norm='none', **linear_kwargs):
        assert len(dims) == num_blocks + 1, "Dims length must be num_blocks + 1."
        super().__init__()
        for i in range(num_blocks):
            self.add_module(
                f'linear_block_{i}',
                LinearBlock(act, norm, in_features=dims[i], out_features=dims[i + 1], **linear_kwargs)
            )

