#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""

import torch
import torch.nn.functional as F
import re
import os
import os.path as P
import nibabel as nib
from . import transforms as tfm
from .metrics import accuracy, dice, sensitivity, precision, specificity, auc, \
sizeL, sizeP, num_false_negative, num_false_positive, num_positive, is_float
from .misc import merge_label_basic


class SegMetricMC(object):
    '''Multi-Class Segmentation Metric Calculator with Print Functions'''
    def __init__(self, nClass=2, sn_list=None, metric_names=None, is_cc=False, n_dim=3):
        self.ACC = 'acc'
        self.DSC = 'dsc'
        self.SENS = 'sens'
        self.SPEC = 'spec'
        self.SIZEL = 'sizeL' # Lesion size
        self.SIZEP = 'sizeP' # Predicted lesion size
        self.FPL = 'fpl'
        self.FNL = 'fnl'
        self.TOTALL = 'totall'
        self.ALL_METRIC = (self.ACC, self.DSC, self.SENS, self.SPEC) #, self.SIZEL, self.SIZEP)
        self.nClass = nClass
        self.is_cc = is_cc
        self.n_dim = n_dim
        if is_cc:
            self.ALL_METRIC = self.ALL_METRIC + (self.FPL, self.FNL, self.TOTALL)
        
        self.calculator = {
                self.ACC: accuracy,
                self.DSC: dice,
                self.SENS: sensitivity,
                self.SPEC: specificity,
                self.SIZEL: sizeL,
                self.SIZEP: sizeP,
                self.FPL: num_false_positive,
                self.FNL: num_false_negative,
                self.TOTALL: num_positive,
                }
        if metric_names == None:
            self.metric_names = self.ALL_METRIC
        else:
            for m in metric_names:
                if m not in self.ALL_METRIC:
                    raise RuntimeError('Unknown specified metric type: %s' % m)
            self.metric_names = metric_names
        
        self.sn_list = sn_list if sn_list else []
        self.buffer = dict()
        self.metric = dict()
        for m in self.metric_names:
            self.buffer[m] = []
            self.metric[m] = 0
            for i in range(nClass):
                self.buffer[m+'/%d'%i] = []
                self.metric[m+'/%d'%i] = 0
        self.buffer_changed = True
        
    def _buffer2metric(self, is_strict=True):
        if self.buffer_changed:
            if len(self) == 0:
                self.buffer_changed = False
                return
            if is_strict:
                assert len(self.sn_list) == len(self.buffer[self.metric_names[0]+'/0']), \
                'Unmatch: lengths of sn_list and buffer.'
            for m in self.metric_names:
                self.metric[m] = float(torch.stack(self.buffer[m]).mean())
                for i in range(self.nClass):
                    self.metric[m+'/%d'%i] = float(torch.stack(self.buffer[m+'/%d'%i]).mean())
        self.buffer_changed = False
        return self.metric
        
    def get_metric(self):
        return self._buffer2metric()
    
    def write_csv(self, epoch, fid):
        '''Write metrics to a csv file mainly for machine to read'''
        self._buffer2metric()
        metric = [str(epoch)]
        for k, v in self.metric.items():
            if k in (self.SIZEL, self.SIZEP):
                continue
            metric.append('%.4f' % v)
        fid.write(', '.join(metric) + '\n')
    
    def write_metric(self, fid, preline=None, is_indiv=False):
        """Write (final) detailed metrics to file"""
        self._buffer2metric()
        if preline:
            fid.write(preline + '\n')
        # Construct total_line and title_line
        metric_str = []
        title_line = '|%20s|' % 'SN'
        for k, v in self.metric.items():
            title_line += '%8s|' % str.upper(k)
            if is_float(v):
                s = '%s = %.4f' % (k, v)
            else:
                s = '%s = %d' % (k, v)
            metric_str.append(s)
        total_line = ', '.join(metric_str)
        fid.write(total_line + '\n')
        if is_indiv:
            fid.write(title_line + '\n')
            # Construct individual lines
            for i, sn in enumerate(self.sn_list):
                line = '|%20s|' % sn
                for _, v in self.buffer.items():
                    w = v[i]
                    if is_float(w):
                        s = '%8.4f|' % w
                    else:
                        s = '%8d|' % w
                    line += s
                fid.write(line + '\n')
        
    def print_metric(self, preword=None, is_indiv=False):
        """Print (final, detailed) metrics to stdout"""
        self._buffer2metric()
        if preword:
            print('%s Segmentation Metrics:' % preword)
        else:
            print('Segmentation Metrics:')
        # Construct total_line and title_line
        metric_str = []
        title_line = '|%20s|' % 'SN'
        for k, v in self.metric.items():
            title_line += '%8s|' % str.upper(k)
            if k is not list(self.metric.keys())[0] and re.match(r'^[^/]*$', k):
                metric_str[-1] += '\n'
            if is_float(v):
                s = '%s = %.4f' % (k, v)
            else:
                s = '%s = %d' % (k, v)
            metric_str.append(s)
        total_line = ', '.join(metric_str)
        print(total_line)
        if is_indiv:
            print(title_line)
            # Construct individual lines
            for i, sn in enumerate(self.sn_list):
                line = '|%20s|' % sn
                for _, v in self.buffer.items():
                    w = v[i]
                    if is_float(w):
                        s = '%8.4f|' % w
                    else:
                        s = '%8d|' % w
                    line += s
                print(line)

    def evaluate_append(self, seg_out, label, sn=None, multilabel_fusetype=None):
        self.buffer_changed = True
        if sn is not None:  # if sn_list also needs to be appended
            if len(self.sn_list) != len(self.buffer[self.metric_names[0]+'/0']):# '/0' for dsc/0
                raise RuntimeWarning('SN is specified but the lengths of sn_list and buffer do not match.')
            else:
                self.sn_list.append(sn)
        if seg_out.ndim == label.ndim:  # multi-class, i.e., each pixel may have multiple labels
            assert seg_out.shape == label.shape, 'pred shape should match label shape: pred %s vs label %s' % (seg_out.shape, label.shape)
            pred = (torch.sigmoid(seg_out) >= 0.5).int()
            if multilabel_fusetype:
                pred = merge_label_basic(pred, multilabel_fusetype)
        else:
            _, pred = torch.max(seg_out, dim=0)
        for m in self.metric_names:
            calc_key = re.match(r'[^/]*', m).group(0) # 'dsc/0' -> 'dsc'
            assert calc_key in self.calculator.keys(), 'Invalid calculator key!'
            temp_metric = []
            for i in range(self.nClass):
                if seg_out.ndim == label.ndim:
                    seg = pred[i]
                    gt = label[i]
                else:
                    seg = (pred==i).int()
                    gt = (label==i).int()
                try:
                    v = self.calculator[calc_key](seg, gt)
                except:
                    print('calculating with CPU')
                    v = self.calculator[calc_key](seg.cpu(), gt.cpu())
                self.buffer[m+'/%d'%i].append(v.cpu())
                temp_metric.append(v)
            # save mean metric value to buffer[m], ignore the background class
            if seg_out.ndim == label.ndim:
                mean = torch.Tensor(temp_metric).mean()
            else:
                mean = torch.Tensor(temp_metric[1:]).mean()
            self.buffer[m].append(mean)
        return pred
        
    def append(self, sn, metrics):
        self.buffer_changed = True
        self.sn_list.append(sn)
        for i, m in enumerate(metrics):
            self.buffer[self.ALL_METRIC[i]].append(m)
            
    def __len__(self):
        return len(self.buffer[self.metric_names[0]+'/0']) # '/0' for dsc/0


def validate_seg(model, dataloader, sn_list, device, num_mo=1, nClass=3, save_dir=None, 
                 is_cc=False, sn_fn_dict=None, patch_size=64, overlap=16, 
                 restore_shape_func=None, restore_infokw=None, 
                 merge_label_func=None, multilabel_fusetype=None):
    """Evaluate multi-ouput model's seg performance on specified dataloader

    Arguements:
        Output: a SegMetric array, elements in the array
        correspond to multiple model outputs.
    """
    sm, sn_counter = [], []
    mo_ind = list(range(-num_mo, 0))
    for _ in mo_ind:
        sm.append(SegMetricMC(nClass, sn_list, is_cc=is_cc))
        sn_counter.append(-1)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        model.to(device)
        model.eval()
        for images, masks in iter(dataloader):
            # print(f'Processing {sn_list[sn_counter[0]+1]}')
            masks = masks.to(device)
            patch_list = tfm.image_to_patch3d(images, patch_size, overlap)
            pred_list = []
            for patch in patch_list:
                pred_list.append(model(patch.to(device)))
                del patch
            preds_seg = tfm.patch_to_image3d(images, pred_list, patch_size, overlap) #.to(device)
            for i in mo_ind: # for each of the multiple outputs (heads)
                for j in range(len(preds_seg[i])): # j indicates each subject in a mini-batch
                    idx = sn_counter[i] = sn_counter[i] + 1
                    seg_out = preds_seg[i, j]
                    seg_mask = sm[i].evaluate_append(seg_out, masks[j],
                                                     multilabel_fusetype=multilabel_fusetype)
                    if save_dir and i == -1:
                        assert sn_fn_dict, 'Please specify SN to filename mapping.'
                        sn = sn_list[idx]
                        if merge_label_func:
                            # print(merge_label_func)
                            seg_mask = merge_label_func(seg_mask, multilabel_fusetype)
                        nii = nib.load(sn_fn_dict[sn])
                        seg = seg_mask.cpu().numpy()
                        if restore_shape_func:
                            seg = restore_shape_func(seg, **restore_infokw[sn])
                        nii_seg = nib.Nifti1Image(seg, nii.affine, nii.header, nii.extra)
                        nii_seg.set_data_dtype('uint16')
                        nii_seg.to_filename(P.join(save_dir, '%s.nii.gz' % sn))
            del seg_mask, seg_out, images, masks, preds_seg, patch_list
    for s in sm:
        s.get_metric()
    
    return sm

def inference(model, dataloader, sn_list, device, save_dir=None, 
              sn_fn_dict=None, patch_size=64, overlap=16, suffix='_seg',
              restore_shape_func=None, restore_infokw=None, 
              merge_label_func=None, multilabel_fusetype=None):
    assert sn_fn_dict, 'Please specify SN to filename mapping.'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        print ('No save directory specified for final true test inference!')
        return
    sn_counter = -1
    with torch.no_grad():
        model.to(device)
        for images, _ in iter(dataloader):
            images = images.to(device)
            patch_list = tfm.image_to_patch3d(images, patch_size, overlap)
            pred_list = []
            for patch in patch_list:
                pred_list.append(model(patch))
            preds_seg = tfm.patch_to_image3d(images, pred_list, patch_size, overlap)
            for j in range(len(preds_seg[-1])): # j indicates each subject in a mini-batch
                seg_out = preds_seg[-1, j]
                if merge_label_func:
                    seg_mask = F.sigmoid(seg_out)>=0.5
                    seg_mask = merge_label_func(seg_mask, multilabel_fusetype)
                else:
                    _, seg_mask = torch.max(seg_out, dim=0)
                    
                sn_counter += 1
                sn = sn_list[sn_counter]
                nii = nib.load(sn_fn_dict[sn])
                seg = seg_mask.cpu().numpy()
                if restore_shape_func:
                    seg = restore_shape_func(seg, **restore_infokw[sn])
                nii_seg = nib.Nifti1Image(seg, nii.affine, nii.header, nii.extra)
                nii_seg.set_data_dtype('uint16')
                nii_seg.to_filename(P.join(save_dir, '%s%s.nii.gz'%(sn, suffix)))

