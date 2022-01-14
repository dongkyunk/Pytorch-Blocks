import torch.nn as nn
from einops import rearrange
from pytorch_blocks.activation import get_activation
from pytorch_blocks.normalization import get_normalization

class ConvBlock(nn.Module):
    """Convolutional block with activation and normalization.

    Args:
        act (str): activation function name.
        norm (str): normalization function name.
        **conv_kargs: keyword arguments for nn.Conv2d.
    """    
    def __init__(self, act='none', norm='none', **conv_kargs):        
        assert norm in ['bn2d', 'ln', 'in2d', 'none'], "Unsupported image normalization function."
        super().__init__()
        self.conv = nn.Conv2d(**conv_kargs)
        self.act = get_activation(act)
        if norm in ['bn2d', 'in2d']:
            self.norm = get_normalization(norm, num_features=conv_kargs['out_channels'])
        elif norm == 'ln':
            # Layer normalization is done on the channel dimension only
            self.norm = get_normalization(norm, normalized_shape=conv_kargs['out_channels'])
            self.reshape = True
        elif norm == 'none':
            self.norm = get_normalization(norm)

    def forward(self, x):
        x = self.conv(x)
        if self.reshape:
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.norm(x)
            x = rearrange(x, 'b h w c-> b c h w')
        else:
            x = self.norm(x)
        x = self.act(x)
        return x


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


