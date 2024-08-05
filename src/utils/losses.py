#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .misc import one_hot


class HybridLoss(object):
    def __init__(self, loss_inst1, loss_inst2, weight=(1,1)):
        self.loss1 = loss_inst1
        self.loss2 = loss_inst2
        self.weight = weight
        
    def __call__(self, input, target):
        # in case of single output
        return self.weight[0]*self.loss1(input, target) + self.weight[1] * self.loss2(input, target)

class WeightedBCEWithLogitsLoss(object):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', 
                 pos_weight=None):
        self.loss = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)
    def __call__(self, input, target):
        return self.loss(input.permute(0,2,3,4,1), target.permute(0,2,3,4,1))


def general_dice_loss(input, target, weight=None, size_average=True, reduce=True, 
                      power=2, ignore_bkg=True, ignore_const=False):
    # input: (n, nClass, d, h, w) logits. target: (n, d, h, w) in [0, nClass-1]
    eps = 1e-6 # stablization constant
    input = F.softmax(input, dim=1)
#    n = input.size(0)
    nClass = input.size(1)
    target_oh = one_hot(target, nClass)
    if ignore_const:
        mbatch_weight = []
        for t in target:
            mbatch_weight.append(0. if t.min()==1 else 1.)
        mbatch_weight = torch.tensor(mbatch_weight, dtype=input.dtype, device=input.device)
    if weight == 'adaptive':
        weight = []
        for c in range(nClass):
            weight.append(1 / max( (target_oh[:,c].sum().float())**power, 25 ))
    elif weight is None:
        weight = [1] * nClass
    weight = torch.tensor(weight, dtype=input.dtype, device=input.device, requires_grad=False)
    if ignore_bkg:
        weight[0] = 0
    numerator = 2 * ((input*target_oh).permute(0,2,3,4,1) * weight).sum(dim=[1,2,3,4]) + eps
    denominator = ((input+target_oh).permute(0,2,3,4,1) * weight).sum(dim=[1,2,3,4]) + eps
    loss = 1 - numerator / denominator # (n,)
    if ignore_const:
        loss = loss * mbatch_weight
    if reduce:
        if size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
    return loss

class GeneralDiceLoss(object):
    def __init__(self, weight=None, size_average=True, reduce=True, power=2, ignore_bkg=True, ignore_const=False):
        self.weight = weight
        self.power = power
        self.size_average = size_average
        self.reduce = reduce
        self.ignore_bkg = ignore_bkg
        self.ignore_const = ignore_const
        
    def __call__(self, input, target):
        return general_dice_loss(input, target, self.weight, self.size_average, self.reduce, 
                                 self.power, self.ignore_bkg, self.ignore_const)

class MultiLabelDiceLoss(object):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average
        
    def __call__(self, input, target):
        assert input.shape == target.shape, 'input shape should match target shape: input %s vs target %s' % (input.shape, target.shape)
        n, c = input.shape[:2]
#        if self.weight is None:
#            weight = [1] * c
#        else:
#            weight = self.weight
#            assert len(weight) == c, 'len(weight) = %d vs nClass = %d' % (len(weight), c)
        if self.weight == 'adaptive':
            weight = []
            for c in range(c):
                weight.append(1 / max( (target[:,c].sum().float())**2, 25 ))
        elif self.weight is None:
            weight = [1] * c
        else:
            weight = self.weight
        weight = [w/sum(weight)*c for w in weight] # sum(weight) == c
        weight = torch.tensor(weight, dtype=input.dtype, device=input.device, requires_grad=False)
        pred = torch.sigmoid(input)
        loss = 0
        for i in range(n):
            for j in range(c):
                loss += weight[j] * (1 - calc_dice(pred[i,j], target[i,j])).to(dtype=input.dtype)
        if self.size_average:
            loss /= n
        return loss

def calc_dice(input, target):
    eps = 1e-6 #Variable(torch.Tensor([1e-6]).cuda(input.get_device()))
#    mask = target >= 0
#    input = input[mask]
#    target = target[mask]
    dice = (2 * (input*target.float()).sum() + eps) / (target.sum().float() + input.sum() + eps)
    return dice

class MultiOutputLoss(object):
    def __init__(self, loss_func, loss_weight, device, decay_factor=1):
        self.loss_func = loss_func
        self.loss_weight = torch.FloatTensor(loss_weight).to(device)
        self.decay_factor = decay_factor
        
    def __call__(self, input, target):
        # in case of single output
        if len(input) == 1:
            loss = self.loss_func(input[0], target)
            loss_arr = [loss]
            return loss, loss_arr
        # in case of multiple output
        loss = 0
        loss_arr = []
        for i in range(len(input)):
            loss_so = self.loss_func(input[i], target)
            loss_arr.append(loss_so)
            loss += self.loss_weight[i] * loss_so
        return loss, loss_arr
        
    def change_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight
                
    def decay_loss_weight(self, decay_factor=None):
        df = decay_factor if decay_factor else self.decay_factor
        for i in range(len(self.loss_weight)-1):
            self.loss_weight[i] *= df
            
    def decay_loss_weight_epoch(self, epoch, decay_factor=None):
        df = decay_factor if decay_factor else self.decay_factor
        for i in range(len(self.loss_weight)-1):
            self.loss_weight[i] *= df
            

class FocalLoss(object):
    def __init__(self, gamma=2, weight=None, size_average=True):
        if isinstance(weight, (list, tuple)):
            weight = torch.Tensor(weight)
        self.weight = weight
        self.size_average = size_average
        self.gamma = gamma
        
    def __call__(self, input, target):
        log_p = F.log_softmax(input, dim=1)
        log_p = torch.pow((1-torch.exp(log_p)), self.gamma) * log_p
        loss = F.nll_loss(log_p, target, weight=self.weight, size_average=False)
        if self.size_average:
            loss /= (target >= 0).data.sum().float()
        return loss
