import torch.nn as nn
from torchvision.ops import stochastic_depth
from pytorch_blocks.conv_block import ConvBlock
from pytorch_blocks.activation import get_activation


class ResBlock(nn.Module):
    """A block with residual connection.

    Args:
        **stochastic_depth_kargs: keyword arguments for torchvision.ops.stochastic_depth.
    """

    def __init__(self, **stochastic_depth_kargs):
        super().__init__()
        self.layers = None
        self.activation = None
        self.stochastic_depth_args = stochastic_depth_kargs

    def forward(self, x):
        skip = x
        x = self.layers(x)
        x = stochastic_depth(x, **self.stochastic_depth_args)
        x = x + skip
        x = self.activation(x)
        return x


class StandardResBlock(ResBlock):
    """Standard residual block in "Deep residual learning for image recognition"    
    https://arxiv.org/abs/1512.03385.

    Args:
        in_channels (int)
        hidden_channels (int)
        act (str): activation function name. Defaults to 'relu'.
        norm (str): normalization function name. Defaults to 'bn'.
        **stochastic_depth_kargs: keyword arguments for torchvision.ops.stochastic_depth.
    """

    def __init__(self, in_channels, hidden_channels, act='relu', norm='bn', **stochastic_depth_kargs):
        super().__init__(**stochastic_depth_kargs)
        self.layers = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=hidden_channels,
                      kernel_size=3, padding='same', act=act, norm=norm),
            ConvBlock(in_channels=hidden_channels, out_channels=in_channels,
                      kernel_size=3, padding='same', act='none', norm=norm),
        )
        self.activation = get_activation(act)


class BottleneckResBlock(ResBlock):
    """Bottleneck residual blocks with 1x1, 3x3, 1x1 convolutions.     
    Args:
        in_channels (int)
        hidden_channels (int)
        act (str): activation function name. Defaults to 'relu'.
        norm (str): normalization function name. Defaults to 'bn'.
        cardinality (int): number of groups for 3x3 convolution. Defaults to 1.
        **stochastic_depth_kargs: keyword arguments for torchvision.ops.stochastic_depth.
    """

    def __init__(self, in_channels, hidden_channels, act='relu', norm='bn', cardinality=1, **stochastic_depth_kargs):
        super().__init__(**stochastic_depth_kargs)
        assert hidden_channels % cardinality == 0
        self.layers = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=hidden_channels,
                      kernel_size=1, padding='same', act=act, norm=norm),
            ConvBlock(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=3, padding='same', groups=cardinality, act=act, norm=norm),
            ConvBlock(in_channels=hidden_channels, out_channels=in_channels,
                      kernel_size=1, padding='same', act='none', norm=norm),
        )
        self.activation = get_activation(act)


class InvertedResBlock(BottleneckResBlock):
    """Inverted residual block in "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    https://arxiv.org/abs/1801.04381

    Args:
        in_channels (int)
        hidden_channels (int)
        act (str): activation function name. Defaults to 'relu'.
        norm (str): normalization function name. Defaults to 'bn'.
        **stochastic_depth_kargs: keyword arguments for torchvision.ops.stochastic_depth.
    """

    def __init__(self, in_channels, hidden_channels, act='relu', norm='bn', **stochastic_depth_kargs):
        assert(hidden_channels >= in_channels,
               "Inverted Residual Block: hidden_channels are usually greater or equal to in_channels")
        super().__init__(in_channels, hidden_channels, act, norm,
                         cardinality=hidden_channels, **stochastic_depth_kargs)


class ResNeXtBlock(BottleneckResBlock):
    """ResNeXt block in "Aggregated Residual Transformations for Deep Neural Networks"
    https://arxiv.org/abs/1611.05431

    Args:
        in_channels (int)
        hidden_channels (int)
        act (str): activation function name. Defaults to 'relu'.
        norm (str): normalization function name. Defaults to 'bn'.
        cardinality (int): number of groups for 3x3 convolution. Defaults to 32.
        **stochastic_depth_kargs: keyword arguments for torchvision.ops.stochastic_depth.
    """

    def __init__(self, in_channels, hidden_channels, act='relu', norm='bn', cardinality=32, **stochastic_depth_kargs):
        assert(cardinality > 1,
               "ResNext Block: cardinality should be greater than 1")
        assert (hidden_channels % cardinality == 0,
                "ResNext Block: hidden_channels should be divisible by cardinality")
        super().__init__(in_channels, hidden_channels, act, norm,
                         cardinality=cardinality, **stochastic_depth_kargs)


class ConvNeXtBlock(ResBlock):
    """ConvNeXt block in "A ConvNet for the 2020s"
    https://arxiv.org/abs/2201.03545
    
    DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)

    Args:
        in_channels (int)
        hidden_channels (int)
        act (str): activation function name. Defaults to 'gelu'.
        norm (str): normalization function name. Defaults to 'ln'.
        **stochastic_depth_kargs: keyword arguments for torchvision.ops.stochastic_depth.
    """

    def __init__(self, in_channels, hidden_channels, act='gelu', norm='ln', **stochastic_depth_kargs):
        super().__init__(**stochastic_depth_kargs)
        self.layers = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=7, padding='same', groups=in_channels, act='none', norm=norm),
            ConvBlock(in_channels=in_channels, out_channels=hidden_channels,
                      kernel_size=1, padding='same', act=act, norm='none'),
            ConvBlock(in_channels=hidden_channels, out_channels=in_channels,
                      kernel_size=1, padding='same', act='none', norm='none'),
        )
        self.activation = get_activation('none')
