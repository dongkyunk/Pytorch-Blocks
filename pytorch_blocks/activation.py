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
    return getattr(nn, activation_dict[activation])(**kwargs)

