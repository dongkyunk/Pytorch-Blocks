import torch.nn as nn

activation_dict = dict(
    relu='ReLU',
    sigmoid='Sigmoid',
    softmax='Softmax',
    softplus='Softplus',
    lrelu='LeakyReLU',
    tanh='Tanh',
    gelu='GELU',
    swish='SiLU',
    none='Identity',
)


def get_activation(activation, **kwargs):
    """Get nn.Module from activation name.

    Args:
        activation (str): name of activation function.
        **kwargs: keyword arguments for activation function.
    Returns:
        nn.Module
    """    
    assert activation in activation_dict, f'{activation} is not supported.'
    return getattr(nn, activation_dict[activation])(**kwargs)

