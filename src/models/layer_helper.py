"""
Created on 11:44 May 3 2022

@author: rongzhao
"""
import torch
import numpy as np

__all__ = ['RoundDifferentiable', 'project_by_iter', 'discretize', 'random_crop',
           'dumb_crop']


class RoundDifferentiable(torch.autograd.Function):
    """A differetiable round function"""

    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


def discretize(var, num_lvl, lo, hi):
    """ Discretize *var* into [lo, hi] on *num_lvl* levels
    e.g., lo, hi can be -1, 1 for weight and 0, 1 for activation.
    [with STE gradient for BP]
    """
    bit_range = num_lvl - 1
    var = torch.clamp(var, lo, hi)
    delta = (hi - lo) / bit_range
    var_ = (var - lo) / delta
    # with STE gradient for BP
    Qvar = RoundDifferentiable.apply(var_)
    Qvar = Qvar * delta + lo
    return Qvar


def project_by_iter(var, num_lvl, lo=-1., hi=1.):
    """ lo, hi should be -1, 1 for weight and 0, 1 for activation
        return: scaling factor a (a real number)
                quantized tensor b (discrete values in [-1, 1])
    """
    flag_nptensor = False
    if not isinstance(var, torch.Tensor):
        var = torch.from_numpy(var)
        flag_nptensor = True
    max_iter = num_lvl * 100
    var = var.double()
    a = var.abs().mean().item()
    a_prev = -999
    c = 0
    with torch.no_grad():
        while abs(a - a_prev) > 1e-5 and c < max_iter:
            # print(a)
            b = discretize(var / a, num_lvl, lo, hi)
            a_prev = a
            a = ((b * var).sum() / (b * b).sum()).item()
            c += 1
            # torch.cuda.empty_cache()
    if c == max_iter:
        print('abs(a-a_prev) = ', abs(a - a_prev))
        raise RuntimeWarning(f'Exceed maximum iteration ({max_iter}) for alpha optimization in var_init_iter')
    # alpha.data = torch.tensor(a, dtype=torch.float)
    # Qvar = discretize(var / a, num_lvl, lo, hi) * a
    b = discretize(var / a, num_lvl, lo, hi).float()
    if flag_nptensor:
        b = b.detach().cpu().numpy()
    return a, b


def random_crop(x, y, size3d=(90, 108, 84), pad=(1,1,1), stride=(1,1,1)):
    n, c, d, h, w = x.shape
    _, _, dy, hy, wy = y.shape
    d_, h_, w_ = size3d
    # padding size
    p1, p2, p3 = pad
    sd = np.random.randint(p1, d - d_ - p1 + 1)
    sh = np.random.randint(p2, h - h_ - p2 + 1)
    sw = np.random.randint(p3, w - w_ - p3 + 1)
    # if stride == 1
    if stride == (1,1,1):
        return x[..., sd - p1:sd + d_ + p1, sh - p2:sh + h_ + p2, sw - p3:sw + w_ + p3], \
               y[..., sd:sd + d_, sh:sh + h_, sw:sw + w_]

    # if stride == 2
    s0, s1, s2 = stride
    dy_, hy_, wy_ = d_ // s0, h_ // s1, w_ // s2
    sdy = np.random.randint(p1, d // s0 - d_ // s0 - p1)
    shy = np.random.randint(p2, h // s1 - h_ // s1 - p2)
    swy = np.random.randint(p3, w // s2 - w_ // s2 - p3)
    sd, sh, sw = sdy * s0, shy * s1, swy * s2
    return x[..., sd - p1:sd + d_ + p1, sh - p2:sh + h_ + p2, sw - p3:sw + w_ + p3], \
           y[..., sdy:sdy + dy_, shy:shy + hy_, swy:swy + wy_]


def dumb_crop(x, y, size3d=None, kernel=None):
    return x, y

