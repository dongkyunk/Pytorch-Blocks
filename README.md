# Pytorch-Blocks
All sorts of predefined blocks for pytorch

## Installation
```pip install git+https://github.com/dongkyuk/Pytorch-Blocks.git```

## Usage

```python
from pytorch_blocks import *

act_example = get_activation('gelu')
norm_example get_normalization('bn1d')
conv_block_example = ConvBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, act='relu', norm='bn2d')
conv_blocks_example = ConvBlocks(num_blocks=2, dims=[3, 64, 128], kernel_size=3, stride=1, padding=1, act='relu', norm='bn2d')
linear_block_example = LinearBlock(in_features=3, out_features=64, act='relu', norm='bn1d')
linear_blocks_example = LinearBlocks(num_blocks=2, dims=[3, 64, 128], act='relu', norm='bn1d')
```
