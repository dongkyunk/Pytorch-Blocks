import torch.nn as nn
from einops.layers.torch import Rearrange
from pytorch_blocks.activation import get_activation
from pytorch_blocks.normalization import get_normalization

class ConvBlock(nn.Sequential):
    """Convolutional block with activation and normalization.

    Args:
        act (str): activation function name.
        norm (str): normalization function name.
        **conv_kargs: keyword arguments for nn.Conv2d.
    """    
    def __init__(self, act='none', norm='none', **conv_kargs):        
        assert norm in ['bn2d', 'ln', 'in2d', 'none'], "Unsupported image normalization function."
        if norm in ['bn2d', 'in2d']:
            norm = get_normalization(norm, num_features=conv_kargs['out_channels'])
            conv_kargs['bias'] = False
        elif norm == 'ln':
            norm = nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                get_normalization(norm, normalized_shape=conv_kargs['out_channels']), # channel dimension only
                Rearrange('b h w c -> b c h w'),
            )
        elif norm == 'none':
            norm = get_normalization(norm)

        super().__init__(
            nn.Conv2d(**conv_kargs),
            norm,
            get_activation(act),
        )



class ConvBlocks(nn.Sequential):
    """Convolutional blocks with activation and normalization.

    Args:
        num_blocks (int): number of convolutional blocks.
        dims (list): list of input and output channels.
        act (str): activation function name.
        norm (str): normalization function name.
        **conv_kargs: keyword arguments for nn.Conv2d.
    """    
    def __init__(self, num_blocks, dims, act='none', norm='none', **conv_kargs):
        assert len(dims) == num_blocks + 1, "Dims length must be num_blocks + 1."
        super().__init__()
        for i in range(num_blocks):
            self.add_module(
                f'conv_block_{i}',
                ConvBlock(act, norm, in_channels=dims[i], out_channels=dims[i + 1], **conv_kargs)
            )


