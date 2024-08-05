#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""

__all__ = []

import torch
import torch.nn as nn
from collections.abc import Iterable
from .factoryQ import make_nBlocks, make_up_fc, \
    ResBlock_wrapper, get_conv_wrapper, ReLU
from . import factoryQ, factory_blk


class ModelQ(nn.Module):
    def __init__(self):
        super(ModelQ, self).__init__()

    def load_state_dict(self, state_dict, strict=True, init=True):
        super(ModelQ, self).load_state_dict(state_dict, strict)
        if init:
            self.qparam_init()

    def qparam_init(self):
        for module in self.modules():
            if 'QConv' in module.__class__.__name__ and hasattr(module, 'qparam_init'):
                module.qparam_init()

    def perform_quantization(self):
        for module in self.modules():
            if 'QConv' in module.__class__.__name__:
                module.perform_quantization()

    def forward(self, x):
        raise NotImplementedError


def scale_tuple(tup, factor):
    if not isinstance(tup, Iterable):
        return tup * factor
    tmp = []
    for x in tup:
        tmp.append(x * factor)
    return tuple(tmp)


class UResQ(ModelQ):
    def __init__(self, QConv, num_mod, num_classes, depth_config, width_config, dilation_config,
                 init_stride=1, stride=2, drop_rate=0.25, nla=ReLU(True), bn=nn.BatchNorm3d,
                 ds=False, blk_type='pre',
                 q_weight=True, qlvl=8, q_act=True, qlvl_act=8, q_first=None, q_last=None,
                 rb=factory_blk.ResBlockWithType, hetero_param=None,
                 save_mem=False, fuse_bn=False, init_kernel=3, is_infer=False,
                 **kwQ):
        super(UResQ, self).__init__()
        assert len(depth_config) == len(width_config) == len(dilation_config)
        assert len(depth_config) % 2 == 1, 'Can only have odd number of UBlocks'
        if hetero_param is None:
            hetero_param = dict()
        aniso_pool_depth = hetero_param.get('aniso_pool_depth', 99999)
        aniso_pool_stride = hetero_param.get('aniso_pool_stride', (2, 2, 1))
        drop_cut_thres = hetero_param.get('drop_cut_thres', -1)
        ds_depth_limit = hetero_param.get('ds_depth_limit', 99999)
        self.init_stride = init_stride
        # %% Wrapper for Q convolution
        ConvQ = get_conv_wrapper(QConv, q_weight, qlvl, q_act, qlvl_act, **kwQ)
        ResBlockQ = ResBlock_wrapper(Conv=ConvQ, nla=nla, bn=bn, rb=rb, blk_type=blk_type)
        if blk_type == 'mid':
            Downer = factory_blk.MaxDown3dMid
            Upper = factory_blk.LinearUp3dMid
        else:
            Downer = factory_blk.MaxDown3dWithType
            Upper = factory_blk.LinearUp3dWithType
        down_sampler = Downer(kernel=stride, Conv=ConvQ, nla=nla, bn=bn, save_mem=save_mem,
                              is_infer=is_infer, blk_type=blk_type)
        down_sampler_ani = Downer(kernel=aniso_pool_stride, Conv=ConvQ, nla=nla, bn=bn,
                                  save_mem=save_mem,
                                  is_infer=is_infer, blk_type=blk_type)
        up_nla = ReLU(False) if blk_type == 'mid' else nla
        up_sampler = Upper(scale_factor=stride, Conv=ConvQ, nla=up_nla, bn=bn, is_infer=is_infer,
                           blk_type=blk_type)
        up_sampler_ani = Upper(scale_factor=aniso_pool_stride, Conv=ConvQ, nla=up_nla, bn=bn,
                               is_infer=is_infer, blk_type=blk_type)

        if blk_type == 'mid':
            fuser = factory_blk.SumFusionMid(up_sampler, Conv=ConvQ, nla=nla, bn=bn, att=None,
                                             stride=stride, fuse_bn=fuse_bn)
            fuser_ani = factory_blk.SumFusionMid(up_sampler_ani, Conv=ConvQ, nla=nla, bn=bn, att=None,
                                                 stride=aniso_pool_stride, fuse_bn=fuse_bn)

        else:
            fuser = factoryQ.SumFusion(up_sampler, Conv=ConvQ, nla=nla, bn=bn, att=None,
                                       stride=stride, fuse_bn=fuse_bn)
            fuser_ani = factoryQ.SumFusion(up_sampler_ani, Conv=ConvQ, nla=nla, bn=bn, att=None,
                                           stride=aniso_pool_stride, fuse_bn=fuse_bn)
        if q_first:
            ConvFirst = get_conv_wrapper(QConv, q_weight=q_first[0] > 0, qlvl=q_first[0],
                                         q_act=q_first[1] > 0, qlvl_act=q_first[1], **kwQ)
        else:
            ConvFirst = nn.Conv3d
        if q_last:
            ConvLast = get_conv_wrapper(QConv, q_weight=q_last[0] > 0, qlvl=q_last[0],
                                        q_act=q_last[1] > 0, qlvl_act=q_last[1], **kwQ)
        else:
            ConvLast = nn.Conv3d
        # %%
        self.conv0 = nn.Sequential()
        if blk_type == 'pre':  # bn relu conv
            self.conv0.add_module('conv', ConvFirst(num_mod, width_config[0], init_kernel,
                                                     init_stride, (init_kernel - 1) // 2,
                                                     bias=False))
        elif blk_type == 'mid':  # relu conv bn
            self.conv0.add_module('conv', ConvFirst(num_mod, width_config[0], init_kernel,
                                                     init_stride, (init_kernel - 1) // 2,
                                                     bias=False))
            self.conv0.add_module('bn', bn(width_config[0]))
        else:  # conv bn relu
            self.conv0.add_module('conv', ConvFirst(num_mod, width_config[0], init_kernel,
                                                     init_stride, (init_kernel - 1) // 2,
                                                     bias=False))
            self.conv0.add_module('bn', bn(width_config[0]))
            self.conv0.add_module('relu', nla())

        self.u_blocks = nn.Sequential()
        self.trans_downs = nn.Sequential()
        self.trans_ups = nn.Sequential()
        self.classifiers = nn.Sequential()

        for i in range(len(depth_config)):
            dr = drop_rate
            if dr > 0 and width_config[i] < drop_cut_thres:
                dr = min(drop_rate / 2, 0.2)
            self.u_blocks.add_module('UResBlock%d' % (i + 1),
                                     make_nBlocks(depth_config[i], width_config[i],
                                                  width_config[i], dr,
                                                  dilation=dilation_config[i],
                                                  rb=ResBlockQ))

            if i < len(depth_config) // 2:
                if i < aniso_pool_depth:
                    td = down_sampler(width_config[i], width_config[i + 1])
                else:
                    td = down_sampler_ani(width_config[i], width_config[i + 1])
                self.trans_downs.add_module('TransDown%d' % (i + 1), td)
            else:
                # exclude the last block
                if i < len(depth_config) - 1:  # TransUps
                    if i >= len(depth_config) - 1 - aniso_pool_depth:
                        tu = fuser(width_config[i], width_config[i + 1], width_config[i + 1])
                    else:
                        tu = fuser_ani(width_config[i], width_config[i + 1], width_config[i + 1])

                    self.trans_ups.add_module('TransUp%d' % (i + 1), tu)
                    if ds:
                        channel_config = width_config[i + 1:]
                        if ds == 'simple':
                            up_fc = make_up_fc(width_config[i], num_classes, up_times=0,
                                               channel_config=None,
                                               extra_up_scale=scale_tuple(init_stride,
                                                                          2 ** len(channel_config)),
                                               Conv=nn.Conv3d,
                                               up_sampler=Upper(scale_factor=stride,
                                                                Conv=nn.Conv3d, nla=nla,
                                                                bn=bn),
                                               nla=nla, bn=bn)
                        else:
                            up_fc = make_up_fc(width_config[i], num_classes,
                                               channel_config=channel_config,
                                               extra_up_scale=init_stride, Conv=nn.Conv3d,
                                               up_sampler=Upper(scale_factor=stride,
                                                                Conv=nn.Conv3d, nla=nla,
                                                                bn=bn), nla=nla, bn=bn)
                        if len(depth_config) - i <= ds_depth_limit:
                            self.classifiers.add_module('AuxClassifier%d' % (i + 1), up_fc)
                        else:
                            self.classifiers.add_module('AuxClassifier%d' % (i + 1), None)

        self.final_cls = nn.Sequential()
        w = width_config[-1]

        self.final_cls.add_module('cls', ConvLast(w, num_classes, 1, 1, 0))
        if init_stride not in (1, (1, 1), (1, 1, 1)):
            self.final_cls.add_module('extra_up',
                                      nn.Upsample(scale_factor=init_stride, mode='trilinear'))

    # %%
    def forward(self, x, feature_out=False):
        n_blocks = len(self.u_blocks)
        n_updown = len(self.trans_downs)
        feature = self.conv0(x)
        skipx = []
        outs = []
        for i in range(n_blocks):
            feature = self.u_blocks[i](feature)
            if i < n_updown:
                skipx.append(feature)
                feature = self.trans_downs[i](feature)
            elif i < n_blocks - 1:
                if self.classifiers and self.classifiers[i - n_updown]:
                    outs.append(self.classifiers[i - n_updown](feature))
                feature = self.trans_ups[i - n_updown](feature, skipx[-(i - n_updown + 1)])
        if feature_out:
            return feature
        outs.append(self.final_cls(feature))
        return torch.stack(outs, dim=0)

