import os
import os.path as P
import torch.nn as nn
import shutil
import sys
from utils import misc
from dataloader.datahub import DataHub_SEG
from dataloader.datasets import Dataset_SEG, Dataset_SEG_OnDisk
import models
from models import model_blk, factory_blk, factoryQ


def get_data_cube(args):
    # %% Dataset config
    data_info = ''
    round_str = 'round' + args.round
    if args.pretrain:
        assert round_str in args.pretrain, 'round number does not match pretrain model!'

    # Label transformation
    tfm_lambda = None
    merge_label_func = None
    if args.bin_label:
        tfm_lambda = lambda img, label: (img, (label > 0).long())
        data_info += '_BinLabel'

    if args.multi_label:
        if args.multi_label.lower() == 'brats':
            tfm_lambda = lambda img, label: (img, misc.split_label_brats(label))
            merge_label_func = misc.merge_label_brats
            data_info += 'MulLabelBRATS'
        if args.multi_label.lower() == 'lits':
            tfm_lambda = lambda img, label: (img, misc.split_label_lits(label))
            merge_label_func = misc.merge_label_basic
            data_info += 'MulLabelLiTS'

    if args.merge_type:
        data_info += '_Merge_' + args.merge_type

    task = args.task
    if task.lower() == 'brats':
        modalities = ('seg', 'flair', 't1', 't1ce', 't2')
        data_dir = '../data/seg/BRATS2020/train_std_crop' if not args.data_dir else args.data_dir
        split_dir = '../data/seg/BRATS2020/split' if not args.split_dir else args.split_dir
        nMod = args.nMod if args.nMod else 4
        nClass = args.nClass if args.nClass else 4

        if not args.patch_size:
            patch_size = (128, 128, 128)
        overlap = 16
        if merge_label_func:
            balance_mask_func = lambda label: label == 3
        else:
            balance_mask_func = lambda label: label == 3
    elif task.lower() == 'lits':
        modalities = ('seg', 'ct')
        data_dir = '../data/seg/LiTS/train_crop_npy_256' if not args.data_dir else args.data_dir
        split_dir = '../data/seg/LiTS/split' if not args.split_dir else args.split_dir
        nMod = args.nMod if args.nMod else 1
        nClass = args.nClass if args.nClass else 3

        if not args.patch_size:
            patch_size = (128, 128, 64)
        overlap = (16, 16, 16)

        if merge_label_func:
            balance_mask_func = lambda label: label[1] > 0
        else:
            balance_mask_func = lambda label: label == 2  # mask_func_lits
    else:
        raise RuntimeError('Unknown task: %s' % task)

    if args.bin_label:
        nClass = 2
    if args.multi_label:
        nClass -= 1

    # if patch_size is specified
    if args.patch_size:
        if ',' in args.patch_size:
            patch_size = tuple([int(s) for s in args.patch_size.split(',')])
        else:
            patch_size = (int(args.patch_size),) * 3

    print(split_dir)

    dh_kwargs = {
        'data_dir': data_dir,
        'train_split': P.join(split_dir, round_str, 'train.txt'),
        'val_split': P.join(split_dir, round_str, 'val.txt'),
        'test_split': P.join(split_dir, round_str, 'test.txt'),
        'train_batchsize': args.batch_size,
        'test_batchsize': 1,
        'modalities': modalities,
        'access_type': args.access_type,
        'mean': None,
        'std': None,
        'rand_flip': (1, 1, 1),
        'crop_type': args.crop_type,
        'balance_rate': args.balance_rate,
        'balance_mask_func': balance_mask_func,
        'crop_size_img': patch_size,
        'DataSet': Dataset_SEG_OnDisk if args.data_on_disk else Dataset_SEG,
        'num_workers': args.num_workers,
        'sn_fn_file': 'sn_fn.txt',
        'slide_patch_size': patch_size,
        'slide_overlap': overlap,
        'tfm_lambda': tfm_lambda,
        'random_noise_prob': args.random_noise_p,
    }
    data_cube = DataHub_SEG(**dh_kwargs)

    need_restore_shape = True
    if need_restore_shape and task.lower() == 'brats':
        restore_shape_func = misc.restore_crop
        import pickle

        with open(os.path.join(data_dir, 'restore_shape_infokw.pickle'), 'rb') as F:
            restore_infokw = pickle.load(F)
    else:
        restore_shape_func = restore_infokw = None
    data_cube.restore_shape_func = restore_shape_func
    data_cube.restore_infokw = restore_infokw
    data_cube.merge_label_func = merge_label_func
    data_cube.multilabel_fusetype = args.merge_type

    return data_cube, data_info, nMod, nClass, patch_size


def get_model_cube(args, QConv=nn.Conv3d, kwQ={}):
    # I/O channels
    if args.task.lower() == 'brats':
        nMod = args.nMod if args.nMod else 4
        nClass = args.nClass if args.nClass else 4
    elif args.task.lower() == 'lits':
        nMod = args.nMod if args.nMod else 1
        nClass = args.nClass if args.nClass else 3
    if args.bin_label:
        nClass = 2
    if args.multi_label:
        nClass -= 1

    # model type
    if args.model in ('UResQ',):
        Net = model_blk.UResQ
    else:
        raise RuntimeError('Unknown model name: %s' % args.model)

    # initial stride
    if ',' in args.init_stride:
        init_stride = tuple(int(x) for x in args.init_stride.split(','))
    else:
        init_stride = (int(args.init_stride),) * 3

    # ResBlock type
    rb = factory_blk.ResBlockWithType
    # Block type
    blk_type = args.blk
    # the model
    model_info = args.model

    if args.qconv.lower() == 'conv':
        q_weight = q_act = False
        q_first = q_last = None
        qlvl = qlvl_act = None
    else:
        q_weight = args.qlvl_w > 0
        q_act = args.qlvl_a > 0
        qlvl = args.qlvl_w
        qlvl_act = args.qlvl_a if q_act else 256

        q_first = q_last = None
        if args.q_first:
            q_first = [int(x) for x in args.q_first.split(',')]
        if args.q_last:
            q_last = [int(x) for x in args.q_last.split(',')]

    # %% NLA type
    if args.nla.lower() == 'relu':
        nla = factoryQ.ReLU(True)
    elif args.nla.lower() == 'reluf':
        nla = factoryQ.ReLU(False)
    else:
        raise RuntimeError('Unknown NLA name: %s' % args.nla)

    # %% norm type
    norm = args.norm.lower()
    model_info += ('_' + norm.upper())
    if norm == 'bn':
        bn = nn.BatchNorm3d
    else:
        raise NotImplementedError('Norm type should be in BN')

    # drop out
    drop_rate = args.drop_rate

    # %% model shape
    # width
    if args.width:
        width_config = [int(i) for i in args.width.split(',')]
    else:
        width_config = [32, 64, 128, 256, 128, 64, 32]
    # depth
    if args.depth:
        depth_config = [int(i) for i in args.depth.split(',')]
    else:
        depth_config = [1] * len(width_config)
    # dilation
    if args.dilation:
        dilation_config = [int(i) for i in args.dilation.split(',')]
    else:
        dilation_config = [1] * len(width_config)

    hetero_param = {
        'drop_cut_thres': 128,  # num of channels lower than which no dropout is employed
        'ds_depth_limit': 3 if 2 in init_stride else 4  # number of DS path; set to 9999 if N/A
    }
    # heteo-dim up/down-sample [not used]
    if args.hetero_dim:
        hetero_param['aniso_pool_depth'] = 9999 if 2 in init_stride else 4
        hetero_param['aniso_pool_stride'] = (2, 2, 1)

    # the model
    model = Net(QConv, nMod, nClass, depth_config=depth_config, width_config=width_config,
                dilation_config=dilation_config, init_stride=init_stride, stride=2,
                drop_rate=drop_rate, nla=nla, bn=bn, ds=args.ds, blk_type=blk_type,
                q_weight=q_weight, qlvl=qlvl, q_act=q_act, qlvl_act=qlvl_act,
                q_first=q_first, q_last=q_last, hetero_param=hetero_param, rb=rb,
                fuse_bn=True, save_mem=True,
                init_kernel=args.init_kernel, **kwQ)

    if args.ds:
        num_mo = min(hetero_param['ds_depth_limit'], len(depth_config) // 2 + 1)
    else:
        num_mo = 1

    model_cube = {
        'model': model,
        'init_func': misc.weights_init,
        'pretrain': args.pretrain,
        'resume': args.resume,
        'optimizer_list': None,
        'num_mo': num_mo,
        'nClass': nClass,
        'nMod': nMod,
    }

    return model_cube, model_info


def get_snapshot_config(args, model_info, Qinfo, model, data_cube):
    timestr = misc.timestr('mdhm')
    round_str = 'round' + args.round
    task = args.task
    # exp_id and snapshot_root definition
    exp_id = f'{model_info}_{timestr}_{Qinfo}'
    exp_id += args.suffix

    snap_id = 'snap'

    snapshot_root = P.join(P.dirname(__file__), '..', 'exp_ptq', task, snap_id, round_str, exp_id)
    print(f'Snapshot to {snapshot_root}')
    os.makedirs(snapshot_root, exist_ok=True)

    shutil.copy2(os.path.abspath(__file__), P.join(snapshot_root, os.path.basename(__file__)))
    if args.config:
        shutil.copy2(args.config, P.join(snapshot_root, os.path.basename(args.config)))

    with open(P.join(snapshot_root, 'cmd.txt'), 'w+') as F:
        F.write(str(sys.argv) + '\n' + ' '.join(sys.argv) + '\n')
        num_param = misc.get_num_param(model)
        F.write('Number of parameters: %d\n' % num_param)

    snapshot_scheme = {
        'root': snapshot_root,
        'display_interval': None,
        'test_interval': None,
        'snapshot_interval': 999999,
        'is_train': False,
        'is_val': True and bool(data_cube.valloader),
        'is_test': True and bool(data_cube.testloader),
    }
    return snapshot_scheme


def get_conv_class(args):
    # %% QConv parameters
    if args.qconv.lower() == 'conv':
        QConv = nn.Conv3d
        Qinfo = 'FP'
        q_weight = q_act = False
        q_first = q_last = None
        qlvl = qlvl_act = None
        kwQ = {}
    else:
        q_weight = args.qlvl_w > 0
        q_act = args.qlvl_a > 0
        qlvl = args.qlvl_w
        qlvl_act = args.qlvl_a if q_act else 256

        q_first = q_last = None
        if args.q_first:
            q_first = [int(x) for x in args.q_first.split(',')]
        if args.q_last:
            q_last = [int(x) for x in args.q_last.split(',')]

        kwQ = {}
        for attr in dir(args):
            if attr[:4] == 'lwq_':
                kwQ[attr] = getattr(args, attr)

        if q_act and q_weight:
            Qinfo = 'bothQw{}a{}'.format(qlvl, qlvl_act)
        elif q_act:
            Qinfo = 'actQa{}'.format(qlvl_act)
        else:
            Qinfo = 'weightQw{}'.format(qlvl)

        Qinfo = args.qconv + '_' + Qinfo

    # %% qconv choice
    if args.qconv.lower() == 'conv':
        QConv = nn.Conv3d
    elif args.qconv.lower() == 'effq':
        QConv = models.EfficientQConv
    else:
        raise RuntimeError('Unknown QConv name: %s' % args.qconv)

    return QConv, Qinfo, kwQ


