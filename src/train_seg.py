#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rongzhao
"""
import os
import os.path as P
import shutil
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import pytorch_warmup as warmup
from utils import misc
from utils.losses import MultiOutputLoss, FocalLoss, GeneralDiceLoss, \
    HybridLoss, MultiLabelDiceLoss, WeightedBCEWithLogitsLoss
from utils.trainer import Trainer
from definer import get_data_cube, get_model_cube


def train_fp(args):
    is_half = False

    warnings.filterwarnings('ignore', '.*output shape of zoom.*')

    device = torch.device(args.device)
    timestr = misc.timestr('mdhm')

    # default datahub parameters
    weight_seg = weight_seg_dice = None
    task = args.task
    data_cube, data_info, nMod, nClass, patch_size = get_data_cube(args)

    # %% Hyper-parameters
    lr = args.lr
    max_epoch = args.max_epoch

    resume = args.resume

    if args.resume:
        pretInfo = 'resume'
    elif args.pretrain:
        pretInfo = 'pret'
    else:
        pretInfo = 'scratch'

    # %% Data Info

    # %% Model definition
    model_cube, model_info = get_model_cube(args)
    model = model_cube['model']

    round_str = 'round' + args.round

    if args.pretrain or args.resume:
        pth = args.resume if args.resume else args.pretrain
        assert round_str in pth, 'round number does not match pretrain/resume model!'
    else:
        model.qparam_init()
    num_mo = model_cube['num_mo']

    if resume:
        experiment_id = resume.split('/')[-2]
    else:
        experiment_id = model_info + '_FP_' + timestr + '_' + pretInfo

        experiment_id += args.suffix

    snapid = 'snap'
    snapshot_root = P.join(P.dirname(__file__), '..', 'exp_fp', task, snapid, round_str,
                           experiment_id)
    os.makedirs(snapshot_root, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    weight_decay = float(args.weight_decay)
    params = model.parameters()
    optimizer_general = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    optim_list = misc.Optimizer_list([optimizer_general])

    model_cube = {
        'model': model,
        'init_func': misc.weights_init,
        'pretrain': args.pretrain,
        'resume': resume,
        'optimizer_list': optim_list,
        'num_mo': num_mo,
        'nClass': nClass,
        'nMod': nMod,
    }
    # %% LR scheme
    exponent = 0.9
    def poly_lr(iter):
        return max(1 - iter / (len(data_cube.trainloader) * max_epoch), 0) ** exponent
    lr_schedulers = [lrs.LambdaLR(opt, poly_lr) for opt in optim_list]
    if args.pretrain:
        warmup_period = 5 * len(data_cube.trainloader)
    else:
        warmup_period = len(data_cube.trainloader)
    warmup_schedulers = [warmup.LinearWarmup(opt, warmup_period=warmup_period) for opt in
                         optim_list]
    lr_scheme = {
        'max_epoch': max_epoch,
        'base_lr': lr,
        'lr_scheduler': misc.LR_scheduler_list(lr_schedulers, warmup_schedulers),
    }
    # %% Loss info
    loss_weight = np.array([1/2**i for i in range(num_mo, 0, -1)])
    for i in range(num_mo-3):
        loss_weight[i] = 0
    loss_weight /= loss_weight.sum()
    lw_decay_seg = 1
    if args.loss.lower() == 'ce':
        loss = nn.CrossEntropyLoss(weight=weight_seg)
    elif args.loss.lower() == 'focal':
        loss = FocalLoss(weight=weight_seg)
    elif args.loss.lower() == 'dice':
        loss = GeneralDiceLoss(weight=weight_seg_dice)
    elif args.loss.lower() == 'hybrid':
        loss = HybridLoss(nn.CrossEntropyLoss(weight=weight_seg),
                          GeneralDiceLoss(weight=weight_seg_dice))
    elif args.loss.lower() == 'focalplusdice':
        loss = HybridLoss(FocalLoss(weight=weight_seg), GeneralDiceLoss(weight=weight_seg_dice))
    elif args.loss.lower() == 'bce':
        loss = WeightedBCEWithLogitsLoss(weight=weight_seg)
    elif args.loss.lower() == 'bdice':
        loss = MultiLabelDiceLoss(weight=weight_seg_dice)
    elif args.loss.lower() == 'bhybrid':
        loss = HybridLoss(WeightedBCEWithLogitsLoss(weight=weight_seg),
                          MultiLabelDiceLoss(weight=weight_seg_dice))
    else:
        raise RuntimeError('Unknown loss type: %s' % args.loss)
    criterion_cube = {
        'criterion_seg': MultiOutputLoss(loss,
                                         loss_weight, device, lw_decay_seg),
    }

    shutil.copy2(os.path.abspath(__file__), P.join(snapshot_root, os.path.basename(__file__)))
    if args.config:
        shutil.copy2(args.config, P.join(snapshot_root, os.path.basename(args.config)))
    if not resume:
        with open(P.join(snapshot_root, 'cmd.txt'), 'w+') as F:
            F.write(str(sys.argv) + '\n' + ' '.join(sys.argv) + '\n')
            num_param = misc.get_num_param(model)
            F.write('Number of parameters: %d\n' % num_param)

    snapshot_scheme = {
        'root': snapshot_root,
        'display_interval': args.disp_interval,
        'test_interval': args.test_interval if args.test_interval > max_epoch/20 else max_epoch // 20,
        'snapshot_interval': 999999,
        'is_train': False,
        'is_val': True and data_cube.valloader,
        'is_test': True and data_cube.testloader,
    }

    tbid = 'tboard'
    writer = SummaryWriter(log_dir=P.join(P.dirname(__file__), '..', 'results', task, tbid, round_str, experiment_id))
    writer_cube = {
        'writer': writer,
        'tb_metric': None,
        'tb_loss': None,
    }

    trainer = Trainer(model_cube, data_cube, criterion_cube, writer_cube,
                      lr_scheme, snapshot_scheme, device, is_half=is_half)
    try:
        with torch.autograd.set_detect_anomaly(True):
            trainer.train()
    except KeyboardInterrupt:
        trainer._snapshot(trainer.epoch, name='KeyboardInterrupt', is_optim=True)
        exit(0)
    print('Training complete.')
    # final evaluation
    is_indiv = True
    is_save_nii = args.save_nii
    is_cc = False
    is_true_test = False

    # need_restore_shape = args.save_nii
    # if need_restore_shape:
    #     restore_shape_func = misc.restore_crop
    #     import pickle
    #     data_dir = data_cube.data_dir
    #     with open(os.path.join(data_dir, 'restore_shape_infokw.pickle'), 'rb') as F:
    #         restore_infokw = pickle.load(F)
    # else:
    restore_shape_func = restore_infokw = None

    trainer.test('seg_max', 'seg_max', is_indiv, is_save_nii, is_cc,
                 is_true_test=is_true_test,
                 restore_shape_func=restore_shape_func, restore_infokw=restore_infokw)

    suffix = '%04d' % max_epoch
    trainer.test(suffix, 'seg_' + suffix, is_indiv, is_save_nii, is_cc,
                 is_true_test=is_true_test,
                 restore_shape_func=restore_shape_func, restore_infokw=restore_infokw)
