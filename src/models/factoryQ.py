#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""
import torch.nn as nn


def passthrough(x):
    return x


class PassModule(nn.Module):
    def __init__(self):
        super(PassModule, self).__init__()

    def forward(self, x):
        return x


def ReLU(inplace=True):
    def nla(inp=None):
        if inp is not None:
            return nn.ReLU(inp)
        return nn.ReLU(inplace)

    return nla


class BNNLAConv_3d(nn.Module):
    def __init__(self, inChans, outChans, kernel, stride=1, padding=0, dilation=1, groups=1,
                 bias=False,
                 Conv=nn.Conv3d, bn=nn.BatchNorm3d, nla=ReLU(True), drop_rate=0.0):
        super(BNNLAConv_3d, self).__init__()
        self.bn = bn(inChans)
        self.relu = nla()
        self.do = nn.Dropout3d(drop_rate) if drop_rate else PassModule()
        self.conv = Conv(inChans, outChans, kernel, stride, padding, dilation, groups, bias)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.do(out)
        out = self.conv(out)
        return out


class ConvBNNLA_3d(nn.Module):
    def __init__(self, inChans, outChans, kernel, stride=1, padding=0, dilation=1, groups=1,
                 bias=False,
                 Conv=nn.Conv3d, bn=nn.BatchNorm3d, nla=ReLU(True), drop_rate=0.0):
        super(ConvBNNLA_3d, self).__init__()
        self.do = nn.Dropout3d(drop_rate) if drop_rate > 0 else PassModule()
        self.conv = Conv(inChans, outChans, kernel, stride, padding, dilation, groups, bias)
        self.bn = bn(outChans)
        self.relu = nla()

    def forward(self, x):
        out = self.do(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class NLAConvBN_3d(nn.Module):
    def __init__(self, inChans, outChans, kernel, stride=1, padding=0, dilation=1, groups=1,
                 bias=False,
                 Conv=nn.Conv3d, bn=nn.BatchNorm3d, nla=ReLU(True), drop_rate=0.0):
        super(NLAConvBN_3d, self).__init__()
        self.relu = nla()
        self.do = nn.Dropout3d(drop_rate) if drop_rate > 0 else PassModule()
        self.conv = Conv(inChans, outChans, kernel, stride, padding, dilation, groups, bias)
        self.bn = bn(outChans)

    def forward(self, x):
        out = self.relu(x)
        out = self.do(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


def LinearUp3d(scale_factor=2, lazy_conv=True, nla=ReLU(True), Conv=nn.Conv3d, bn=nn.BatchNorm3d,
               is_infer=False, blk_type='pre'):
    if is_infer:
        raise Warning('Normal LinearUp3d should NOT have is_infer == True.')

    def upsampler(inChans, outChans):
        seq_model = nn.Sequential()
        if inChans == outChans:
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        elif nla != None:
            seq_model.add_module('bn', bn(inChans))
            seq_model.add_module('relu', nla())
            seq_model.add_module('conv', Conv(inChans, outChans, 1, 1, 0, bias=False))
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        else:
            seq_model.add_module('conv', Conv(inChans, outChans, 1, 1, 0, bias=False))
            seq_model.add_module('trilinear',
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
        return seq_model

    return upsampler


def SumFusion(upsampler, Conv=nn.Conv3d, bn=nn.BatchNorm3d, nla=ReLU(True), att=None, stride=2,
              fuse_bn=False, blk_type='pre'):
    class Fuser(nn.Module):
        def __init__(self, inChans, skipChans, outChans):
            super(Fuser, self).__init__()
            if att:
                raise RuntimeError('SumFusion: This version does not support attention.')
            self.upsampler = upsampler(inChans, skipChans)
            self.fuse_bn = fuse_bn
            if fuse_bn:
                self.bn_x = bn(skipChans)
                self.bn_skip = bn(skipChans)

        def forward(self, x, skipx):
            x = self.upsampler(x)
            if self.fuse_bn:
                return self.bn_x(x) + self.bn_skip(skipx)
            return x + skipx

    return Fuser


class ResBlock(nn.Module):
    """3D ResBlock of *full pre-activetion*"""
    def __init__(self, inChans, outChans, drop_rate=0.5, dilation=1, nla=ReLU(True),
                 Conv=nn.Conv3d, bn=nn.BatchNorm3d):
        super(ResBlock, self).__init__()
        self.change_dim = inChans != outChans
        self.bn1 = bn(inChans)
        self.relu1 = nla()
        self.conv1 = Conv(inChans, outChans, 3, 1, dilation, dilation, bias=False)
        self.bn2 = bn(outChans)
        self.relu2 = nla()
        self.conv2 = Conv(outChans, outChans, 3, 1, dilation, dilation, bias=False)
        self.do = passthrough if drop_rate == 0 else nn.Dropout3d(drop_rate)
        self.projection = Conv(inChans, outChans, 1, 1, 0, bias=False) \
            if self.change_dim else passthrough

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        out = self.conv2(self.do(out))
        out += self.projection(x)
        return out


class ResBlockWithType(nn.Module):
    """3D ResBlock with 3 different types *full pre-activetion or ReLU-only pre-activetion
    or ReLU before addition"""
    def __init__(self, inChans, outChans, drop_rate=0.5, dilation=1, nla=ReLU(True),
                 Conv=nn.Conv3d, bn=nn.BatchNorm3d, blk_type='pre'):
        super().__init__()
        self.change_dim = inChans != outChans
        if blk_type == 'pre':
            blk = BNNLAConv_3d
        elif blk_type == 'mid':
            blk = NLAConvBN_3d
        else:
            blk = ConvBNNLA_3d
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


def get_conv_wrapper(QConv, q_weight, qlvl, q_act, qlvl_act, **kw):
    if QConv in (nn.Conv2d, nn.Conv3d):
        return QConv

    def get_conv(in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        return QConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                     bias,
                     q_weight=q_weight, qlvl=qlvl, q_act=q_act, qlvl_act=qlvl_act, **kw)

    return get_conv


def ResBlock_wrapper(Conv=nn.Conv3d, bn=nn.BatchNorm3d, nla=ReLU(True), rb=ResBlock, **kw):
    def get_resblock(inChans, outChans, drop_rate=0.5, dilation=1):
        return rb(inChans, outChans, drop_rate, dilation, nla, Conv, bn, **kw)

    return get_resblock


def make_nBlocks(nBlocks, inChans, outChans, drop_rate=0.5, dilation=1, rb=ResBlock):
    # return pass if nB = 0
    if nBlocks == 0:
        return PassModule()
    # change dimension at the first convolution
    seq_model = nn.Sequential()
    seq_model.add_module('Layer1', rb(inChans, outChans, drop_rate, dilation=dilation))
    for i in range(nBlocks - 1):
        seq_model.add_module('Layer%d' % (i + 2), rb(outChans, outChans, drop_rate, dilation))
    return seq_model


def make_up_fc(in_chans, num_classes, up_times=None, channel_config=None, extra_up_scale=None,
               Conv=nn.Conv3d, up_sampler=LinearUp3d(), nla=ReLU(True), bn=nn.BatchNorm3d):
    seq_model = nn.Sequential()
    if channel_config == up_times == None:
        up_times = 0
    if channel_config is None:
        channel_config = [in_chans // (2 ** i) for i in range(1, up_times + 1)]
    elif up_times is None:
        up_times = len(channel_config)
    else:
        assert len(channel_config) == up_times, 'up_times and len(channel_config) are not equal.'
    if up_times > 0:
        seq_model.add_module('up1', up_sampler(in_chans, channel_config[0]))
    for i in range(up_times - 1):
        seq_model.add_module('up%d' % (i + 2), up_sampler(channel_config[i], channel_config[i + 1]))

    seq_model.add_module('classifier',
                         Conv(channel_config[-1] if up_times > 0 else in_chans, num_classes, 1, 1,
                              0))
    if extra_up_scale is not None and (extra_up_scale not in (1, (1, 1), (1, 1, 1))):
        seq_model.add_module('extra_up', nn.Upsample(scale_factor=extra_up_scale, mode='trilinear'))

    return seq_model

