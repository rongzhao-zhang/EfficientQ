#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score
import nibabel as nib


def is_float(value):
    if isinstance(value, torch.Tensor):
        return value.is_floating_point()
    return isinstance(value, float)

'''All following metrics are calculated as Tensor'''
def dice(pred_b, target_b):
    '''Calculate Dice index of the binary prediction'''
    eps = 1e-6
    dice_index = (2*(pred_b*target_b).sum().float() + eps) / (pred_b.sum().float() + target_b.sum().float() + eps)
    return dice_index

def accuracy(pred_b, target_b):
    '''Calculate accuracy of the binary prediction'''
    accuracy = ((pred_b==target_b).sum().float()) / torch.tensor(target_b.numel(), dtype=torch.float, device=target_b.device)
    return accuracy

def sensitivity(pred_b, target_b):
    '''Calculate sensitivity of the binary prediction'''
    eps = 1e-6
    sensitivity_index = ((pred_b*target_b).sum().float() + eps) / (target_b.sum().float() + eps)
    return sensitivity_index

def specificity(pred_b, target_b):
    '''Calculate specificity of the binary prediction'''
    device = pred_b.device
    zero, one = torch.tensor(0, device=device), torch.tensor(1, device=device)
    eps = 1e-6
    pred_bn = torch.where(pred_b>0, zero, one)
    target_bn = torch.where(target_b>0, zero, one)
    specificity_index = ((pred_bn*target_bn).sum().float() + eps) / (target_bn.sum().float() + eps)
    return specificity_index

def sizeL(pred_b, target_b):
    return target_b.sum().float()

def sizeP(pred_b, target_b):
    return pred_b.sum().float()

def precision(pred_b, target_b):
    '''Calculate precision of the binary prediction'''
    eps = 1e-6
    prec = ((pred_b*target_b).sum().float() + eps) / (pred_b.sum().float() + eps)
    return prec

def auc(prob, target_b):
    '''Calculate area under ROC curve (AUC) score'''
    prob_np = prob.cpu().numpy()
    target_np = target_b.cpu().numpy()
    try:
        return roc_auc_score(target_np, prob_np)
    except ValueError: # In case only one class present
        return 1

def num_component(mask):
    mask_np = mask.cpu().numpy()
    _, num_compo = ndimage.label(mask_np, np.ones((3,3)))
    num_compo = torch.tensor(num_compo, dtype=torch.float32, device=mask.device)
    return num_compo
    
def num_false_positive(pred_b, target_b):
    pred_np = pred_b.cpu().numpy()
    target_np = target_b.cpu().numpy()
    false_counter = 0
    pred_compo, num_compo = ndimage.label(pred_np, np.ones((3,3)))
    for i in range(1, num_compo+1):
        current_pred = np.where(pred_compo == i, 1, 0)
        overlap = target_np * current_pred
        if ~overlap.any():
            false_counter += 1
    return torch.tensor(false_counter, dtype=torch.float32, device=target_b.device)

def num_positive(pred_b, target_b):
    return num_component(target_b)

def num_false_negative(pred_b, target_b):
    return num_false_positive(target_b, pred_b)

def num_true_positive(pred_b, target_b):
    return num_positive(pred_b, target_b) - num_false_negative(pred_b, target_b)

def prob2label(pred):
    '''Retrieve image-wise label from the raw network output (can be of (C,) for CNN, 
    or (C,H,W) for multi-instance learning CNN)'''
    if len(pred.size()) == 1:
        _, label = torch.max(pred, dim=0)
        return label
    pred = pred.view(pred.size(0), -1)
    mini_nega_loc = torch.argmin(pred[0])
    label = torch.argmax(pred[:, mini_nega_loc])
    return label

def raw2prob(pred):
    '''Retrieve image-wise label probability from the raw network output (can be of (C,) for CNN, 
    or (C,H,W) for multi-instance learning CNN)'''
    if len(pred.size()) == 1:
        prob = F.softmax(pred, dim=0)[1]
        return prob
    pred = pred.view(pred.size(0), -1)
    mini_nega_loc = torch.argmin(pred[0])
    prob = F.softmax(pred[:, mini_nega_loc], dim=0)[1]
    return prob


def validate_vs_label(output, target, task='lits'):
    """
    Compute the DICE between FP and Quant outputs
    :param output:
    :param target:
    :param task:
    :return:
    """
    if output.dim() >= 6:  # MNCDHW
        measure_mo = []
        for i in range(len(output)):
            m = validate_vs_label(output[i], target, task)
            measure_mo.append(m)
        return measure_mo
    else:  # NCDHW
        if task == 'lits':
            _, pred = torch.max(output, 1)
            measure = []
            for c in range(output.shape[1]):
                measure.append(dice(pred == c, target == c))
        elif task == 'brats':
            pred = (torch.sigmoid(output) >= 0.5).int()
            print(pred.shape)
            measure = [dice(pred.sum(dim=1) == 0, target.sum(dim=1) == 0)]  # for BKG
            for c in range(output.shape[1]):
                measure.append(dice(pred[:, c], target[:, c]))
        else:
            raise RuntimeError(f'Unknown task {task}')
        # print(f'{task}: {measure}')
        return measure


def mean(nums):
    s = 0
    for n in nums:
        s += n
    return s / len(nums)


def print_metric(measure):
    num_mo = len(measure)
    n_classes = len(measure[0])
    all_msg = ''
    for i in range(num_mo):
        msg = f'output{i-num_mo}: '
        for c in range(n_classes):
            msg += f'\tDSC/{c} = {measure[i][c]:.4f}'
        msg += f'\tDSC/fg = {mean(measure[i][1:]):.4f}'
        print(msg)
        all_msg += msg + '\n'
    return all_msg


def get_pred_lits(out: torch.Tensor):
    """

    :param out: has shape NCDHW
    :return: prediction NDHW
    """
    _, pred = torch.max(out, 1)
    return pred


def get_pred_brats(out: torch.Tensor):
    """

    :param out: has shape NCDHW
    :return: prediction NDHW
    """
    hard = (torch.sigmoid(out) >= 0.5)
    pred = torch.zeros_like(hard[:,0]).int()  # NDHW
    for i in range(hard.shape[1]):
        pred[hard[:, i]] = i + 1
    return pred


def get_pred_brats_con_merge(out: torch.Tensor):
    """

    :param out: has shape NCDHW
    :return: prediction NDHW
    """
    hard = (torch.sigmoid(out) >= 0.5).int()
    for ii in range(1, hard.shape[1]):  # conservative merge
        hard[:, ii] = hard[:, ii] * hard[:, ii - 1]

    pred = torch.zeros_like(hard[:, 0]).int()  # NDHW
    for i in range(hard.shape[1]):
        pred[hard[:, i] > 0] = i + 1

    return pred


def extract_nii(multi_output: torch.Tensor, task='lits'):
    out = multi_output[-1]
    if task == 'lits':
        pred = get_pred_lits(out)
    elif task == 'brats':
        pred = get_pred_brats_con_merge(out)
    else:
        raise RuntimeError(f'Unknown task {task}')
    niis = []
    for i in range(len(out)):  # for each in the mini-batch
        img = pred[i].detach().cpu().numpy().astype('uint8')
        nii = nib.Nifti1Image(img, np.eye(4))
        niis.append(nii)
    return niis

