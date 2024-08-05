import torch
import torch.nn as nn
from .factoryQ import BNNLAConv_3d, NLAConvBN_3d, ConvBNNLA_3d, PassModule, ReLU


"""
factory functions for different resblock types
"""


blk_map = {
    'pre': BNNLAConv_3d,
    'mid': NLAConvBN_3d,
    'post': ConvBNNLA_3d,
}


def MaxDown3dMid(kernel=2, stride=None, padding=0, dilation=1, return_indices=False,
              ceil_mode=False, lazy_conv=False, Conv=nn.Conv3d, nla=ReLU(True), bn=nn.BatchNorm3d,
              save_mem=False, is_infer=False, blk_type='mid'):
    if stride is None:
        stride = kernel

    def down(inChans, outChans):
        seq_model = nn.Sequential()
        if nla is not None:
            blk = blk_map[blk_type]
            seq_model.add_module('pool',
                                 nn.MaxPool3d(kernel, stride, padding, dilation, return_indices,
                                              ceil_mode))
            # seq_model.add_module('norm', bn(inChans))
            seq_model.add_module('block', blk(inChans, outChans, 1, 1, 0, 1,
                                              bias=False, Conv=Conv, bn=bn, nla=nla, drop_rate=0))
        else:
            seq_model.add_module('conv', Conv(inChans, outChans, 1, 1, 0, bias=False))
            seq_model.add_module('pool',
                                 nn.MaxPool3d(kernel, stride, padding, dilation, return_indices,
                                              ceil_mode))

        return seq_model

    return down


def LinearUp3dMid(scale_factor=2, lazy_conv=True, nla=ReLU(True), Conv=nn.Conv3d, bn=nn.BatchNorm3d,
               is_infer=False, blk_type='mid'):
    if is_infer:
        raise Warning('Normal LinearUp3d should NOT have is_infer == True.')

    def upsampler(inChans, outChans):
        seq_model = nn.Sequential()
        blk = blk_map[blk_type]
        if inChans == outChans:
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        elif nla != None:
            seq_model.add_module('block', blk(inChans, outChans, 1, 1, 0, 1,
                                              bias=False, Conv=Conv, bn=bn, nla=nla, drop_rate=0))
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
            # seq_model.add_module('norm',
            #                      bn(outChans))
        else:
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
            seq_model.add_module('conv', Conv(inChans, outChans, 1, 1, 0, bias=False))
        return seq_model

    return upsampler


def SumFusionMid(upsampler, Conv=nn.Conv3d, bn=nn.BatchNorm3d, nla=ReLU(True), att=None, stride=2,
              fuse_bn=False, blk_type='mid'):
    class Fuser(nn.Module):
        def __init__(self, inChans, skipChans, outChans):
            super(Fuser, self).__init__()
            if att:
                raise RuntimeError('SumFusion: This version does not support attention.')
            self.upsampler = upsampler(inChans, skipChans)
            # self.bn = bn(skipChans)
            # self.fuse_bn = fuse_bn
            # if fuse_bn:
            #     self.bn_x = bn(skipChans)
            #     self.bn_skip = bn(skipChans)

        def forward(self, x, skipx):
            x = self.upsampler(x)
            # if self.fuse_bn:
            #     return self.bn_x(x) + self.bn_skip(skipx)
            # return self.bn(x + skipx)
            return x + skipx

    return Fuser


def MaxDown3dWithType(kernel=2, stride=None, padding=0, dilation=1, return_indices=False,
              ceil_mode=False, lazy_conv=False, Conv=nn.Conv3d, nla=ReLU(True), bn=nn.BatchNorm3d,
              save_mem=False, is_infer=False, blk_type='pre'):
    if stride is None:
        stride = kernel

    def down(inChans, outChans):
        seq_model = nn.Sequential()
        if nla is not None:
            seq_model.add_module('pool',
                                 nn.MaxPool3d(kernel, stride, padding, dilation, return_indices,
                                              ceil_mode))
            blk = blk_map[blk_type]
            seq_model.add_module('block', blk(inChans, outChans, 1, 1, 0, 1,
                                              bias=False, Conv=Conv, bn=bn, nla=nla, drop_rate=0))
        else:
            seq_model.add_module('conv', Conv(inChans, outChans, 1, 1, 0, bias=False))
            seq_model.add_module('pool',
                                 nn.MaxPool3d(kernel, stride, padding, dilation, return_indices,
                                              ceil_mode))

        return seq_model

    return down


def LinearUp3dWithType(scale_factor=2, lazy_conv=True, nla=ReLU(True), Conv=nn.Conv3d, bn=nn.BatchNorm3d,
               is_infer=False, blk_type='pre'):
    if is_infer:
        raise Warning('Normal LinearUp3d should NOT have is_infer == True.')

    def upsampler(inChans, outChans):
        seq_model = nn.Sequential()
        blk = blk_map[blk_type]
        if inChans == outChans:
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        elif nla != None:
            seq_model.add_module('block', blk(inChans, outChans, 1, 1, 0, 1,
                                              bias=False, Conv=Conv, bn=bn, nla=nla, drop_rate=0))
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        else:
            seq_model.add_module('conv', Conv(inChans, outChans, 1, 1, 0, bias=False))
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        return seq_model

    return upsampler


class ResBlockWithType(nn.Module):
    """3D ResBlock with 3 different types *full pre-activetion or ReLU-only pre-activetion
    or ReLU before addition"""
    def __init__(self, inChans, outChans, drop_rate=0.5, dilation=1, nla=ReLU(True),
                 Conv=nn.Conv3d, bn=nn.BatchNorm3d, blk_type='pre'):
        super().__init__()
        self.change_dim = inChans != outChans
        blk = blk_map[blk_type]
        self.block1 = blk(inChans, outChans, 3, 1, dilation, dilation, bias=False,
                          Conv=Conv, bn=bn, nla=nla, drop_rate=0)
        self.block2 = blk(outChans, outChans, 3, 1, dilation, dilation, bias=False,
                          Conv=Conv, bn=bn, nla=nla, drop_rate=drop_rate)
        self.projection = Conv(inChans, outChans, 1, 1, 0, bias=False) \
            if self.change_dim else PassModule()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out + self.projection(x)
        return out


