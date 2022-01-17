# Pytorch-Blocks
All sorts of predefined blocks for pytorch

## Installation
```pip install git+https://github.com/dongkyuk/Pytorch-Blocks.git```

## Usage

### Activation, Normalization
```python
from pytorch_blocks import get_activation

get_activation('gelu')
get_normalization('bn1d')
```

### Linear Blocks
```python
from pytorch_blocks import LinearBlock, LinearBlocks

LinearBlock(in_features=3, out_features=64, act='relu', norm='bn1d')
LinearBlocks(num_blocks=2, dims=[3, 64, 128], act='relu', norm='bn1d')
```

### Convolutional Blocks
```python
from pytorch_blocks import ConvBlock, ConvBlocks

ConvBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, act='relu', norm='bn2d')
ConvBlocks(num_blocks=2, dims=[3, 64, 128], kernel_size=3, stride=1, padding=1, act='relu', norm='bn2d')
```

### Residual Block Variants

For all residual block variants, you can use stochastic depth with a few arguments.

Normalization and Activation Functions can also be passed as arguments.

```python
from pytorch_blocks import *

StandardResBlock(in_channels=256, hidden_channels=64, act='relu', norm='bn', use_stochastic_depth=True, p=1)
BottleneckResBlock(in_channels=256, hidden_channels=64)
InvertedResBlock(in_channels=64, hidden_channels=256)
ResNeXtBlock(in_channels=256, hidden_channels=128, cardinality=32)
ConvNeXtBlock(in_channels=64, hidden_channels=256)
SEResBlock(in_channels=256, hidden_channels=64)
SEBottleneckResBlock(in_channels=256, hidden_channels=64)
```
