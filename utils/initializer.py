import numpy as np
import torch
import torch.nn as nn

'''
    Copied and modified from
    https://github.com/oxwhirl/treeqn/blob/ae885feb85d3ae4e23a8ade415ceae747b668968/treeqn/utils/pytorch_utils.py
'''

def ortho_init(tensor, scale=1.0):

    shape = tensor.size()
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4:
        flat_shape = (shape[0] * shape[2] * shape[3], shape[1])  # NCHW
    else:
        raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    w = (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    tensor.copy_(torch.FloatTensor(w))
    return tensor


def nn_init(module, w_init=ortho_init, w_scale=1.0, b_init=nn.init.constant_, b_scale=0.0):
    w_init(module.weight.data, w_scale)
    b_init(module.bias.data, b_scale)
    return module


def xav_init(module):
    nn.init.xavier_normal_(module.weight)
    return module
