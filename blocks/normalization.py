import torch.nn as nn

normalization_dict = dict(
    bn1d='BatchNorm1d',
    bn2d='BatchNorm2d',
    in1d='InstanceNorm1d',
    in2d='InstanceNorm2d',
    ln='LayerNorm',
    none='Identity',
)


def get_normalization(normalization, **kwargs):
    return getattr(nn, normalization_dict[normalization])(**kwargs)
