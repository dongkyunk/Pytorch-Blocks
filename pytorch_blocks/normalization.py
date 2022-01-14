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
    """Get nn.Module from normalization name.

    Args:
        normalization (str): name of normalization function.
        **kwargs: keyword arguments for normalization function.
    Returns:
        nn.Module
    """    
    return getattr(nn, normalization_dict[normalization])(**kwargs)
