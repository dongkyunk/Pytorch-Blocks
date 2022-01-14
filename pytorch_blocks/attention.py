import torch.nn as nn
from einops import rearrange
from pytorch_blocks.linear_block import LinearBlock


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block.

    Args:
        in_channels (int)
        reduction (int): reduction factor for hidden channels. Defaults to 16.
        hidden_channels (int): number of hidden channels. Defaults to in_channels // reduction.
        act (str): activation function name. Defaults to 'relu'.
    """    
    def __init__(self, in_channels, reduction=16, hidden_channels=None, act='relu'):
        super().__init__()
        hidden_channels = in_channels // reduction
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            LinearBlock(in_features=in_channels, out_features=hidden_channels, act=act),
            LinearBlock(in_features=hidden_channels, out_features=in_channels, act='sigmoid'),
        )

    def forward(self, x):
        y = self.layers(x)
        y = rearrange(y, 'b c -> b c h w', h=x.shape[2], w=x.shape[3])
        y = x * y
        return y

