#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""

import sys
# sys.path.append("..")  # Adds higher directory to python modules path.
import os
import os.path as P
import torch
import torch.nn as nn
import numpy as np
import time, datetime

from .losses import MultiOutputLoss
from .misc import timestr
from . import misc
from .validate import validate_seg, inference


class Trainer(object):
    """A functional class facilitating and supporting all procedures in training phase"""

    def __init__(self, model_cube, data_cube, criterion_cube, writer_cube,
                 lr_scheme, snapshot_scheme, device, wrap_test=False, is_half=False):
        self.device = device
        if is_half:
            self.dtype = torch.half
        else:
            self.dtype = torch.float

        pretrain = model_cube.get('pretrain')
        resume = model_cube.get('resume')

        self.model = model_cube.get('model')
        self.optimizer_list = model_cube.get('optimizer_list')
        self.num_mo = model_cube.get('num_mo')
        self.nClass = model_cube.get('nClass')
        self.nMod = model_cube.get('nMod')
        self.parse_dataloader(data_cube)
        self.data_cube = data_cube
        self.parse_criterion(criterion_cube)
        self.parse_writer(writer_cube)
        self.lr_scheme = lr_scheme
        self.lr_scheduler = lr_scheme.get('lr_scheduler')
        self.max_epoch = lr_scheme.get('max_epoch') if not wrap_test else 0
        self.snapshot_scheme = snapshot_scheme
        self.root = snapshot_scheme.get('root')
        self.device = device
        self.iter = 0
        self.epoch = 0

        if not wrap_test:
            # resume or load pretrain
            self.start_epoch = 1
            self.max_seg_metric_val = 0
            if resume:
                if P.isfile(resume):
                    # resume model
                    self.start_epoch = self._resume(resume)
                    self.iter = self.start_epoch * len(data_cube.trainloader)
                else:
                    raise RuntimeError('No checkpoint found in %s' % resume)
            elif pretrain:
                if os.path.isfile(pretrain):
                    self._load_pretrain(self.model, pretrain)
                else:
                    raise RuntimeError('No checkpoint found at %s' % pretrain)
            else:
                weight_init_func = model_cube.get('init_func')
                self.model.apply(weight_init_func)

        if not wrap_test and not resume:
            with open(P.join(self.root, 'description.txt'), 'w') as f:
                f.write(str(lr_scheme) + '\n' + str(snapshot_scheme) + '\n' + str(self.model))

        if wrap_test:
            with open(P.join(self.root, 'description_test.txt'), 'w') as f:
                f.write(str(lr_scheme) + '\n' + str(snapshot_scheme) + '\n' + str(self.model))

        self.model.to(self.device, self.dtype)

    def train(self):
        """Cordinate the whole training phase, mainly recording of losses and metrics, 
        lr and loss weight decay, snapshotting, etc."""
        loss_all = []
        # max_seg_metric_val = max_seg_metric_tr = 0
        is_metric_new = False
        lossF = open(P.join(self.root, 'loss.txt'), 'a')
        seg_metricF = open(P.join(self.root, 'seg_metric.txt'), 'a')
        print(timestr(), 'Optimization Begin')
        start_time = time.time()
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            # Adjust learning rate
            loss_dict = self.train_epoch()
            loss_all.append(loss_dict['loss'])
            self.epoch = epoch
            if epoch % self.snapshot_scheme['display_interval'] == 0 or epoch == self.start_epoch:
                N = self.snapshot_scheme['display_interval']
                loss_avg = np.array(loss_all[-N:]).mean()
                first_epoch = epoch if epoch == self.start_epoch else epoch + 1 - N
                ellapse_time = time.time() - start_time
                est_time_length = ellapse_time / (epoch - self.start_epoch + 1) * (self.max_epoch - self.start_epoch)
                est_finish_time = misc.datetime_from_now(est_time_length - ellapse_time)
                print('%s Epoch %d~%d: loss = %.5f, lr = %.5e. End:%s, total: %s' %
                      (timestr(), first_epoch, epoch, loss_avg, self._get_lr(), est_finish_time,
                       str(datetime.timedelta(seconds=int(est_time_length)))))
                lossF.write('%d,%.7f\n' % (epoch, loss_avg))
                lossF.flush()

            if epoch % self.snapshot_scheme['snapshot_interval'] == 0 or epoch == self.start_epoch:
                self._snapshot(epoch, is_optim=True)

            if epoch % self.snapshot_scheme['test_interval'] == 0 or epoch == self.start_epoch:
                metric_dict = self.validate_online(epoch, seg_metricF)
                is_metric_new = True
                metric_seg_val = metric_dict['val/seg_dsc']
                self._snapshot(epoch, 'latest', is_optim=True)

                if self.max_seg_metric_val < metric_seg_val and epoch > 10:
                    self.max_seg_metric_val = metric_seg_val
                    self._snapshot(epoch, 'seg_max', is_optim=True)

            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                if self.tb_loss:
                    for k in self.tb_loss:
                        self.writer.add_scalar(k, float(loss_dict[k]), epoch)
                else:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(k, float(v), epoch)
                if is_metric_new:
                    if self.tb_metric:
                        for k in self.tb_metric:
                            self.writer.add_scalar(k, float(metric_dict[k]), epoch)
                    else:
                        for k, v in metric_dict.items():
                            self.writer.add_scalar(k, float(v), epoch)
                    is_metric_new = False

            self.criterion.decay_loss_weight()

        self._snapshot(self.max_epoch, is_optim=True)

        self._final_snap('FP')
        if self._final_quantization():
            self._final_snap('Qtz')

        seg_metricF.close()
        lossF.close()
        misc.try_remove(P.join(self.root, 'state_0001.pkl'))
        misc.try_remove(P.join(self.root, 'state_current.pkl'))
        misc.try_remove(P.join(self.root, 'state_latest.pkl'))
        misc.try_remove(P.join(self.root, 'state_KeyboardInterrupt.pkl'))

    @staticmethod
    def fuse_agg(pred):
        """Aggressive multilabel fusion for single output (CDHW).
           This is a mutator (will modify input)"""
        predx = torch.empty_like(pred)
        predx[-1] = pred[-1]
        for i in range(len(pred) - 2, -1, -1):
            predx[i] = torch.max(pred[i], pred[i + 1])
        return predx

    @staticmethod
    def fuse_con(pred):
        """Conservative multilabel fusion for single output (CDHW).
           This is a mutator (will modify input)"""
        predx = torch.empty_like(pred)
        predx[0] = pred[0]
        for i in range(1, len(pred)):
            predx[i] = torch.min(pred[i], pred[i - 1])
        return predx

    @staticmethod
    def fuse_agg_batch(batch_out):
        """Aggressive multilabel fusion for mini-batch output (NCDHW).
           This is a mutator (will modify input)"""
        outx = torch.empty_like(batch_out)
        for i, pred in enumerate(batch_out):
            outx[i] = Trainer.fuse_agg(pred)
        return outx

    @staticmethod
    def fuse_con_batch(batch_out):
        """Conservative multilabel fusion for mini-batch output (NCDHW).
           This is a mutator (will modify input)"""
        outx = torch.empty_like(batch_out)
        for i, pred in enumerate(batch_out):
            outx[i] = Trainer.fuse_con(pred)
        return outx

    @staticmethod
    def multilabel_fuse(seg_output, fusetype, criterion):
        if fusetype:
            if fusetype.lower() in ('agg', 'aggressive'):
                if isinstance(criterion, MultiOutputLoss):
                    segx = torch.empty_like(seg_output)
                    for i, batch_out in enumerate(seg_output):
                        segx[i] = Trainer.fuse_agg_batch(batch_out)
                else:
                    segx = Trainer.fuse_agg_batch(seg_output)
            elif fusetype.lower() in ('con', 'conservative'):
                if isinstance(criterion, MultiOutputLoss):
                    segx = torch.empty_like(seg_output)
                    for i, batch_out in enumerate(seg_output):
                        segx[i] = Trainer.fuse_con_batch(batch_out)
                else:
                    segx = Trainer.fuse_con_batch(seg_output)
            else:
                raise RuntimeError('Unknown Multilabel Fusetype: %s' % fusetype)
            return segx
        return seg_output

    def train_epoch(self):
        """Train the model for one epoch, loss information is recorded"""
        self.model.train()
        loss_buf = []
        loss_arr_buf = []
        for images, masks in iter(self.trainloader):
            images, masks = images.to(self.device, self.dtype), masks.to(self.device)
            self.optimizer_list.zero_grad()
            seg_output = self.model(images)
            if torch.isnan(seg_output).any():
                print('NaN')
            loss_seg, loss_seg_arr = self.criterion(seg_output, masks)

            loss_seg.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 1)
            self.optimizer_list.step()
            loss_buf.append(loss_seg.item())
            loss_arr_buf.append(torch.tensor(loss_seg_arr, device='cpu').detach().numpy())
            # For big epochs: adjust LR and record LR and Loss
            self.iter += 1
            self.lr_scheduler.step()
            self.writer.add_scalar('LR_iter', self._get_lr(), self.iter)
            self.writer.add_scalar('Loss_iter', loss_seg, self.iter)
            # For big epoch: logging loss and lr
            if self.epoch <= 2 and self.iter % self.snapshot_scheme['display_interval'] == 0:
                N = self.snapshot_scheme['display_interval']
                loss_avg = np.array(loss_buf[-N:]).mean()
                first_iter = self.iter if self.iter == 0 else self.iter + 1 - N
                print('%s Iter %d ~ %d: loss = %.7f, current lr = %.7e' %
                      (timestr(), first_iter, self.iter, loss_avg, self._get_lr()))

        loss_dict = self.format_loss_buffer(loss_buf, loss_arr_buf)

        return loss_dict

    def test(self, state_suffix, save_dir, is_indiv=False, is_save_nii=False,
             is_cc=False, is_true_test=False):
        """Coordinate the testing of the model after training"""
        save_dir = P.join(self.root, save_dir)
        if state_suffix:
            pretrain = P.join(self.root, 'state_%s.pkl' % state_suffix)
            self._load_pretrain(self.model, pretrain, init=False)
        self.validate_final(save_dir, is_indiv, is_save_nii, is_cc)
        if is_true_test:
            self.inference_final(P.join(save_dir, 'true_test'), '')

    def test_given_pretrain(self, pretrain, save_dir, is_indiv=False, is_save_nii=False, is_cc=False,
                            is_true_test=False):
        """Cordinate the testing of the model after training"""
        save_dir = P.join(self.root, save_dir)
        self._load_pretrain(self.model, pretrain, init=False)
        self.validate_final(save_dir, is_indiv, is_save_nii, is_cc)
        if is_true_test:
            self.inference_final(P.join(save_dir, 'true_test'))

    def validate_final(self, save_dir, is_indiv, is_save_nii,
                       is_cc=False):
        """Validate the model after training finished, detailed metrics would be recorded"""
        def validate_split(dataloader, sn_list, split):
            nii_dir = P.join(save_dir, split) if is_save_nii else None
            sm_arr = \
                validate_seg(self.model, dataloader, sn_list, self.device, self.num_mo,
                             self.nClass, nii_dir, is_cc=is_cc, sn_fn_dict=self.sn_fn_dict,
                             patch_size=self.slide_patch_size, overlap=self.slide_overlap,
                             restore_shape_func=self.data_cube.restore_shape_func,
                             restore_infokw=self.data_cube.restore_infokw,
                             merge_label_func=self.data_cube.merge_label_func,
                             multilabel_fusetype=self.data_cube.multilabel_fusetype)

            split_segF = open(P.join(save_dir, '%s_seg.txt' % split), 'w')
            for i in range(-1, -self.num_mo - 1, -1):
                sm = sm_arr[i]
                sm.write_metric(split_segF, 'Output %d:' % i, is_indiv)
            split_segF.close()
            sm_arr[-1].print_metric('  ' + split)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        if self.snapshot_scheme['is_train'] and self.trainseqloader:
            # switch train_dataset to fix transform mode
            self.trainseqloader.dataset.use_fix_transform()
            validate_split(self.trainseqloader, self.train_sn, 'train')
            # switch train_dataset back to random transform mode
            self.trainseqloader.dataset.use_random_transform()
        if self.snapshot_scheme['is_val'] and self.valloader:
            validate_split(self.valloader, self.val_sn, 'val')
        if self.snapshot_scheme['is_test'] and self.testloader:
            validate_split(self.testloader, self.test_sn, 'test')

    def inference_final(self, save_dir, suffix):
        inference(self.model, self.true_test_image_loader, self.true_test_sn, self.device,
                  save_dir=P.join(self.root, save_dir), sn_fn_dict=self.sn_fn_dict,
                  patch_size=self.slide_patch_size, overlap=self.slide_overlap, suffix=suffix,
                  restore_shape_func=self.data_cube.restore_shape_func,
                  restore_infokw=self.data_cube.restore_infokw,
                  merge_label_func=self.data_cube.merge_label_func,
                  multilabel_fusetype=self.data_cube.multilabel_fusetype)

    def validate_online(self, epoch, seg_metricF):
        """Validate the model during training, record a minimal number of metrics"""

        def validate_split(dataloader, sn_list, split):
            sm_arr = \
                validate_seg(self.model, dataloader, sn_list, self.device, self.num_mo,
                             nClass=self.nClass, save_dir=None,
                             patch_size=self.slide_patch_size, overlap=self.slide_overlap,
                             merge_label_func=self.data_cube.merge_label_func,
                             multilabel_fusetype=self.data_cube.multilabel_fusetype)

            sm_arr[-1].print_metric('  ' + split)
            sm_arr[-1].write_csv(epoch, seg_metricF)
            seg_metricF.flush()
            return sm_arr

        metric_dict = dict()

        def convert_metric_dict(prefix, sm_arr):
            """Convert metrics in the last output"""
            for k, v in sm_arr[-1].metric.items():
                metric_dict['%s/seg_%s' % (prefix, k)] = v

        if self.snapshot_scheme['is_train'] and self.trainseqloader:
            # switch train_dataset to fix transform mode
            self.trainseqloader.dataset.use_fix_transform()
            sm_arr_train = \
                validate_split(self.trainseqloader, self.train_sn, 'Train')
            convert_metric_dict('train', sm_arr_train)
            # switch train_dataset back to random transform mode
            self.trainseqloader.dataset.use_random_transform()
        if self.snapshot_scheme['is_val'] and self.valloader:
            sm_arr_val = \
                validate_split(self.valloader, self.val_sn, 'Validation')
            convert_metric_dict('val', sm_arr_val)
        if self.snapshot_scheme['is_test'] and self.testloader:
            sm_arr_test = \
                validate_split(self.testloader, self.test_sn, 'Test')
            convert_metric_dict('test', sm_arr_test)

        return metric_dict

    @staticmethod
    def format_loss_buffer(loss_buf, loss_arr_buf):
        """Gather all different losses into a dictionary with clear-named keys"""
        loss = np.array(loss_buf).mean()
        loss_arr = np.array(loss_arr_buf).mean(axis=0)
        loss_dict = dict()
        loss_dict['loss'] = loss
        for i in range(-len(loss_arr), 0):
            loss_dict['loss/%d' % i] = loss_arr[i]
        return loss_dict

    def parse_dataloader(self, data_cube):
        self.trainloader = data_cube.trainloader
        self.valloader = data_cube.valloader
        self.testloader = data_cube.testloader
        self.trainseqloader = data_cube.trainseqloader
        self.val_sn = data_cube.val_sn
        self.test_sn = data_cube.test_sn
        self.train_sn = data_cube.train_sn
        self.true_test_sn = data_cube.true_test_sn
        self.sn_fn_dict = data_cube.sn_to_fn_map
        self.true_test_image_loader = data_cube.true_test_image_loader
        self.slide_patch_size = data_cube.slide_patch_size
        self.slide_overlap = data_cube.slide_overlap

    def parse_criterion(self, criterion_cube):
        if criterion_cube is None:
            return
        self.criterion = criterion_cube.get('criterion_seg', None)

    def parse_writer(self, writer_cube):
        if writer_cube is None:
            return
        self.writer = writer_cube.get('writer', None)
        self.tb_metric = writer_cube.get('tb_metric', None)
        self.tb_loss = writer_cube.get('tb_loss', None)

    @staticmethod
    def _load_pretrain(model, pretrain, strict=False, init=True):
        """meaning of init: whether perform qparam_init()"""
        state = torch.load(pretrain, 'cpu')
        model.load_state_dict(state['state_dict'], strict, init)

    def _resume(self, pretrain):
        #        raise RuntimeWarning('optimizer_list is not resumed in current code.')
        state = torch.load(pretrain, 'cpu')
        if 'max_metric' in state:
            print('resuming max metric ... ')
            self.max_seg_metric_val = state['max_metric']
        if 'state_dict' in state:
            print('resuming model weights ... ')
            self.model.load_state_dict(state['state_dict'], init=False)

        if 'optimizer_list' in state:
            print('resuming optimizer ... ')
            self.optimizer_list.load_state_dict(state['optimizer_list'])
            self.optimizer_list.to(self.device)
        else:
            print('No saved optimizer')
        if 'lr_state' in state:
            print('resuming lr scheduler ...')
            self.lr_scheduler.load_state_dict(state['lr_state'])
        else:
            print('No saved lr scheduler, trying to inference from epoch number')
            self.lr_scheduler[0].last_epoch = (state['epoch']) * len(self.trainloader)
        return state['epoch'] + 1

    def _get_lr(self, group=0):
        return self.lr_scheduler[0].get_last_lr()[0]

    def _snapshot(self, epoch, name=None, is_optim=True):
        """Take snapshot of the model, save to root dir"""
        state_dict = {'epoch': epoch,
                      'state_dict': self.model.state_dict(),
                      'lr_state': self.lr_scheduler.state_dict(),
                      'max_metric': self.max_seg_metric_val
                      }
        if is_optim:
            state_dict['optimizer_list'] = self.optimizer_list.state_dict()
        if name is None:
            filename = '%s/state_%04d.pkl' % (self.root, epoch)
        else:
            filename = '%s/state_%s.pkl' % (self.root, name)
        print('%s Snapshotting to %s' % (timestr(), filename))
        torch.save(state_dict, filename)

    def _optim_device(self, device):
        self.optimizer_list.to(device)

    @staticmethod
    def _decay_beta(model, betapow):
        for name, module in model.named_modules():
            if module.__class__.__name__ == 'TerAct':
                module.beta *= betapow

    def _final_quantization(self):
        flag = False
        for name, module in self.model.named_modules():
            if 'QConv' in module.__class__.__name__:
                module.perform_quantization()
                flag = True
        return flag

    def _restore_from_final_snap(self, fname, strict=False, init=True):
        sdx = np.load(fname, allow_pickle=True)['state_dict'].item()
        state_dict = {}
        for k, v in sdx.items():
            if 'conv.weight' in k and len(misc.make_set(v.numpy())) <= 3:
                k_alpha = k.split('.')[:-1].join('.') + 'alpha_w'
                v = v.float() * sdx[k_alpha]
            state_dict[k] = v
        self.model.load_state_dict(state_dict, strict, init)

    def _final_snap(self, name):
        self.model.cpu()
        state_dict = self.model.state_dict()
        sdx = {}
        for k, v in state_dict.items():
            if 'conv.weight' in k and len(misc.make_set(v.numpy())) <= 3:
                v /= v.max()
                v = v.type(torch.int8)
            sdx[k] = v
        filename = '%s/state_%s' % (self.root, name)
        np.savez_compressed(filename, state_dict=sdx)
        torch.save(sdx, filename + '_torch.pkl')
