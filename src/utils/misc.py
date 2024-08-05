#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:52:38 2017

@author: zhang
"""
import time, datetime, pytz
import numpy as np
import os
import os.path as P
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
from scipy import ndimage
from collections.abc import Iterable

__all__ = ['timestr', 'bilinear_init2d', 'weights_init', 'get_names']


def timestr(form=None):
    dt = datetime.datetime.now()
    cn = pytz.timezone('PRC')
    dtcn = dt.astimezone(cn)
    if form is None:
        return dtcn.strftime("<%Y-%m-%d %H:%M:%S>")
    if form == 'mdhm':
        return dtcn.strftime('%m%d%H%M')


def datetime_from_now(seconds, form=None):
    dt = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    cn = pytz.timezone('PRC')
    dtcn = dt.astimezone(cn)
    if form is None:
        return dtcn.strftime("<%m-%d %H:%M>")
    if form == 'mdhm':
        return dtcn.strftime('%m%d%H%M')


def bilinear_init3d(m, method=nn.init.kaiming_normal):
    inC, outC, d, h, w = m.weight.size()
    if not (outC == 1 and inC == m.groups and h == w == 4):
        method(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    else:
        kernel = torch.zeros(d, h, w)
        f_d = math.ceil(d / 2.)
        f_h = math.ceil(h / 2.)
        f_w = math.ceil(w / 2.)
        c_d = (2 * f_d - 1 - f_d % 2) / (2. * f_d)
        c_h = (2 * f_h - 1 - f_h % 2) / (2. * f_h)
        c_w = (2 * f_w - 1 - f_w % 2) / (2. * f_w)
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    kernel[i, j, k] = (1 - math.fabs(i / f_d - c_d)) * \
                                      (1 - math.fabs(j / f_h - c_h)) * (1 - math.fabs(k / f_w - c_w))
        for i in range(inC):
            m.weight.data[i] = kernel


def bilinear_init2d(m, method=nn.init.kaiming_normal):
    inC, outC, h, w = m.weight.size()
    if not (outC == 1 and inC == m.groups and h == w == 4):
        method(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    else:
        kernel = torch.zeros(h, w)
        f_h = math.ceil(h / 2.)
        f_w = math.ceil(w / 2.)
        c_h = (2 * f_h - 1 - f_h % 2) / (2. * f_h)
        c_w = (2 * f_w - 1 - f_w % 2) / (2. * f_w)
        for j in range(h):
            for k in range(w):
                kernel[j, k] = (1 - math.fabs(j / f_h - c_h)) * (1 - math.fabs(k / f_w - c_w))
        for i in range(inC):
            m.weight.data[i] = kernel


def weights_init(m, method=nn.init.kaiming_normal):
    """
    ConvXd: kaiming_normal (weight), zeros (bias)
    BatchNormXd: ones (weight), zeros (bias)
    ConvTransposedXd: bilinear (weight), no bias
    """
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1:
        method(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose3d') != -1:
        bilinear_init3d(m, method)
    elif classname.find('ConvTranspose2d') != -1:
        bilinear_init2d(m, method)


def random_split(file_list, split_nums, seed=None):
    import random
    random.seed(seed)
    file_list = file_list.copy()
    random.shuffle(file_list)
    total = sum(split_nums)
    splits_accum = [round(sum(split_nums[:i]) / total * len(file_list)) for i in range(len(split_nums) + 1)]
    splits = [file_list[splits_accum[i]:splits_accum[i + 1]] for i in range(len(split_nums))]
    for s in splits:
        s.sort()
    random.seed(None)
    return splits


def list_join(*l):
    jl = []
    for li in l:
        jl += li
    return jl


def cross_validation_random_split(file_list, num_rounds, num_splits, seed=None):
    import random
    random.seed(seed)
    file_list = file_list.copy()
    random.shuffle(file_list)
    n = len(file_list)
    segments = []
    num_per_seg = n // num_rounds
    for i in range(num_rounds - 1):
        segments.append(file_list[i * num_per_seg:(i + 1) * num_per_seg])
    segments.append(file_list[(i + 1) * num_per_seg:])
    rounds = []
    for i in range(num_rounds):
        splits = []
        splits.append(list_join(*segments[:-(num_splits - 1)]))
        splits += segments[-(num_splits - 1):]
        for s in splits:
            s.sort()
        rounds.append(splits)
        segments = segments[1:] + segments[0:1]
    random.seed(None)
    return rounds


def file_to_dict(fname, sep=','):
    if fname is None:
        return None
    with open(fname, 'r') as f:
        lines = f.read().splitlines()
    d = dict()
    for line in lines:
        k, v = line.split(sep)
        d[k] = v
    return d


def restore_crop(crop, pmin, pmax, shape):
    '''Restore the shape of a cropped 3D image'''
    image = np.zeros(shape, dtype=crop.dtype)
    if len(shape) == 3:
        image[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = crop
    elif len(shape) == 2:
        image[pmin[0]:pmax[0], pmin[1]:pmax[1]] = crop
    else:
        raise RuntimeError('Invalid restore shape:', list(shape))
    return image


def get_names(iter_num):
    if type(iter_num) == int:
        pklname = 'state_%04d.pkl' % iter_num
        foldername = 'pred_%d' % iter_num
    else:
        pklname = 'state_%s.pkl' % iter_num
        foldername = 'pred_%s' % iter_num
    return pklname, foldername


class Optimizer_list(object):
    def __init__(self, optim_list):
        self.optimizer_list = optim_list

    def step(self):
        for optim in self.optimizer_list:
            optim.step()

    def zero_grad(self):
        for optim in self.optimizer_list:
            optim.zero_grad()

    def state_dict(self):
        sd_list = [optim.state_dict() for optim in self.optimizer_list]
        return sd_list

    def load_state_dict(self, sd_list):
        for sd, optim in zip(sd_list, self.optimizer_list):
            optim.load_state_dict(sd)

    def __getitem__(self, ind):
        return self.optimizer_list[ind]

    def __len__(self):
        return len(self.optimizer_list)

    def __iter__(self):
        return iter(self.optimizer_list)

    def to(self, device):
        for optimizer in self.optimizer_list:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)


def merge_label_basic(label, multilabel_fusetype=None):
    pred = label
    if multilabel_fusetype.lower() in ('agg', 'aggressive'):
        for i in range(len(pred)):
            pred[i] = sum(pred[i:]) > 0  # if p[j]==1 for any j>i, then p[i]==1
    elif multilabel_fusetype.lower() in ('con', 'conservative'):
        for i in range(1, len(pred)):
            pred[i] = pred[i] * pred[i - 1]  # if p[j]==0 for any j<i, then p[i]==0
    else:
        raise RuntimeError('Unknown Multilabel Fusetype: %s' % multilabel_fusetype)
    return pred


def merge_label_brats(label, multilabel_fusetype=None):
    """label[0] for whole tumor (WT), label[1] for tumor core (TC), label[2] for enhancing tumor (ET)
       CxDxHxW -> DxHxW"""
    _, d, h, w = label.size()
    label = label.int()
    if multilabel_fusetype:
        label = merge_label_basic(label, multilabel_fusetype)
    merged = torch.zeros(d, h, w, dtype=label.dtype, device=label.device)  # CxDxHxW -> DxHxW
    merged[label[0] != 0] = 1  # Whole Tumor (WT)
    merged[(label[0] != 0) * (label[1] == 0)] = 2  # ED = WT - TC
    merged[label[2] != 0] = 4  # ET = ET
    return merged


def merge_label_brats_inference(label, multilabel_fusetype=None):
    merged = merge_label_brats(label, multilabel_fusetype)
    et = (merged == 4).cpu().numpy()
    compo, num_compo = ndimage.label(et)
    for i in range(1, num_compo + 1):
        if (et == i).sum() > 500:  # result et indicates components that are smaller than 500
            et[et == i] = 0
    et = torch.from_numpy(et)
    merged[et > 0] = 2  # ET components that are smaller than 500 are labeled as 2 (NCR)
    return merged


def split_label_brats(label):
    split = torch.zeros(3, *label.shape, dtype=torch.float, device=label.device)  # DxHxW -> CxDxHxW
    split[0] = (label > 0)  # WT
    split[1] = ((label == 1) + (label == 3))  # TC
    split[2] = (label == 3)  # ET
    return split


def merge_label_lits(label, multilabel_fusetype=None):
    '''input: label == 0 for liver, label == 1 for tumor
       output: 0 for bkg, 1 for liver, 2 for tumor'''
    c, d, h, w = label.size()
    label = label.int()
    if multilabel_fusetype:
        label = merge_label_basic(label, multilabel_fusetype)
    merged = torch.zeros(d, h, w, dtype=label.dtype, device=label.device)  # CxDxHxW -> DxHxW
    merged[label[0] != 0] = 1  # Liver
    merged[label[1] != 0] = 2  # Tumor
    return merged


def split_label_lits(label):
    split = torch.zeros(2, *label.shape, dtype=torch.float, device=label.device)  # DxHxW -> CxDxHxW
    split[0] = (label > 0)  # Liver
    split[1] = (label == 2)  # Tumor
    return split


class LR_scheduler_list(object):
    """ReduceLROnPlateau for a list of optimizers."""

    def __init__(self, lr_schedulers, warm_ups=None):
        self.lr_scheduler_list = lr_schedulers
        self.warm_up_list = warm_ups

    def step(self):
        if self.warm_up_list:
            for scheduler, warm_up in zip(self.lr_scheduler_list, self.warm_up_list):
                scheduler.step()
                warm_up.dampen()
        else:
            for scheduler in self.lr_scheduler_list:
                scheduler.step()

    def __getitem__(self, ind):
        return self.lr_scheduler_list[ind]

    def state_dict(self):
        sd_list = [lrs.state_dict() for lrs in self.lr_scheduler_list]
        return sd_list

    def load_state_dict(self, sd_list):
        for sd, lrs in zip(sd_list, self.lr_scheduler_list):
            lrs.load_state_dict(sd)

    def get_lr(self, ind):
        self.lr_scheduler_list[ind].optimizer.param_groups[0]['lr']


def str_to_tuple(s, spliter=',', n_dim=3):
    if spliter in s:
        tmp = []
        for x in s.split(spliter):
            tmp.append(int(x))
        return tuple(tmp)
    return (int(s),) * n_dim


def get_num_param(model):
    s = 0
    for p in model.parameters():
        s += p.numel()
    return s


def intensity_01(img):
    mi, ma = img.min(), img.max()
    img = (img - mi) / (ma - mi)
    return img


def intensity_gamma(img, gamma):
    img = intensity_01(img)
    return img ** gamma


def make_set(w):
    s = set(list(x for x in w.reshape(-1)))
    return s


def try_remove(fname):
    if P.isfile(fname):
        print('Removing', fname)
        os.remove(fname)


def one_hot(label, nClass, dim=1):
    """Convert label (..., D, H, W) to (..., [C,] D, H, W), C is at dimension [dim]"""
    label_new = []
    for i in range(nClass):
        label_new.append((label == i))
    label_new = torch.stack(label_new, dim=dim).float()
    return label_new
