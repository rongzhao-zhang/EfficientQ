import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from models.PTQConv import PTQConv
from models.PTQBlock import PTQBlock
from models.hooks import forward_hook, backward_hook
from models.fold_bn import search_fold_and_remove_bn
from utils.tester import PTQTester
import time
import os.path as P
from utils.metrics import extract_nii, get_pred_brats, get_pred_lits


# set FP/PTQ flags
def set_fp(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.set_fp()


def set_quantizing(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.set_quantizing()


def set_init_alpha(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv,)):
            module.set_init_act()


def set_quantized(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.set_quantized()


def store_int_weight(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv,)):
            module.store_int_weight()


def restore_fp_weight(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv,)):
            module.restore_fp_weight()


def set_debug(m: nn.Module):
    for module in m.modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.debug = True


def set_name(m: nn.Module):
    for name, module in m.named_modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.name = name


def set_snapdir(m: nn.Module, snap_dir):
    for name, module in m.named_modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.snap_dir = snap_dir


def set_mask(m: nn.Module, mask_pyramid):
    for name, module in m.named_modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            module.mask_pyramid = mask_pyramid


def set_anything(m: nn.Module, attr, a_thing):
    for name, module in m.named_modules():
        if isinstance(module, (PTQConv, PTQBlock)):
            setattr(module, attr, a_thing)


def get_calibration_data(args, data_cube):
    # remember that trainseqloader should be used with fix_transform (and better switch back to
    # rand_transform if trainloader is also in use)
    data_cube.trainseqloader.dataset.use_fix_transform()
    data_iter = iter(data_cube.trainseqloader)

    for _ in range(args.lwq_dataid):
        next(data_iter)

    from dataloader.transforms import center_crop
    if args.lwq_batchsz == 1:
        data_batch, label_batch = next(data_iter)  # the 0-th for (img, label)
        if args.lwq_patchsz:
            crop_shape = [int(x) for x in args.lwq_patchsz.split(',')]
        else:
            crop_shape = [min(x, 192) // 64 * 64 for x in data_batch.shape[-3:]]
        data_batch = center_crop(data_batch, crop_shape)
        label_batch = center_crop(label_batch, crop_shape)
    else:
        crop_shape = [int(x) for x in args.lwq_patchsz.split(',')]
        data_batch, label_batch = [], []
        for _ in range(args.lwq_batchsz):
            datapoint, lab = next(data_iter)
            data_batch.append(center_crop(datapoint, crop_shape))
            label_batch.append(center_crop(lab, crop_shape))
        data_batch = torch.cat(data_batch, dim=0)
        label_batch = torch.cat(label_batch, dim=0)

    return data_batch, label_batch


# this function is not used
def quantize(model, data_batch):
    """

    :param model:
    :param data_batch:
    :return:

    The logic:
    Given: model, calibration data
    DO:
    1) infer the model once, recording both imtermediate and final outputs
    2) quantize the model
    3) backpropagate from the loss between out_fp and out_q, record the gradients
       (wrt both weights and activations, backward hooks)
    4) quantize the model again
    5) if necessary, repeat 3) and 4) for several times
    6) return the quantized model
    """
    # infer once
    out_fp = model(data_batch)[0]  # multiple_output: MNCDHW => NCDHW
    prob_fp_log = F.log_softmax(out_fp, dim=1)

    # quantize for the first time
    l = F.kl_div()


def get_mask_pyramid(output_fp: torch.Tensor, body_mask: torch.Tensor, weight_map: dict, init_stride: str, num_lvls: int = 5,
                     task='lits'):
    if ',' in init_stride:
        init_stride = tuple(int(x) for x in init_stride.split(','))
    else:
        init_stride = (int(init_stride),) * 3

    out = output_fp[-1]  # index -1 for multiple outputs
    out = F.avg_pool3d(out, init_stride)
    body_mask = F.max_pool3d(body_mask.float(), init_stride).bool()
    pyramid = []

    # case 1
    for i in range(num_lvls):
        if task == 'lits':
            pred_fp = get_pred_lits(out)
        elif task == 'brats':
            pred_fp = get_pred_brats(out)
        else:
            raise RuntimeError(f'Unknown task {task}')
        mask = torch.ones_like(pred_fp)
        for k, v in weight_map.items():
            mask[pred_fp == k] = v
        mask[~body_mask] = 1
        pyramid.append(mask.float().cpu())
        out = F.avg_pool3d(out, 2)
        body_mask = F.max_pool3d(body_mask.float(), 2).bool()

    return pyramid


def get_class_num_lits(pred: torch.Tensor, body_mask: torch.Tensor):
    nums = []  # number of voxels of each class
    nClass = 3
    for i in range(nClass):
        nums.append(((pred == i) & body_mask).sum().item())
        # print(f'pred_fp == {i} is {nums[-1]}')
    return nums


def get_class_num_brats(pred, body_mask: torch.Tensor):
    nClass = 4
    nums = [(torch.sum(pred, dim=1) == 0).sum().item() - (~body_mask).sum().item()]  # number of voxels of each class: [bkg]
    # print(f'Class bkg is {nums[-1]}')
    for i in range(nClass - 1):
        nums.append((pred[:, i] * body_mask).sum().item())  # [bkg, WT, TC, ...]
        # print(f'Class {i + 1} is {nums[-1]}')
    return nums


def fuse_con(pred):
    """Conservative multilabel fusion for single output (CDHW).
       This is a mutator (will modify input)"""
    predx = torch.empty_like(pred)  # CDHW
    predx[0] = pred[0]
    for i in range(1, len(pred)):
        predx[i] = torch.min(pred[i], pred[i - 1])
    return predx


def fuse_con_rawout(output):
    M, N = output.shape[:2]
    out = torch.empty_like(output)
    for i in range(M):
        for j in range(N):
            out[i, j] = fuse_con(output[i, j])
    return out


def get_att_weight_map(output_fp: torch.Tensor, body_mask: torch.Tensor, style: str,
                       task: str = 'lits'):
    out = output_fp[-1]  # index -1 for multiple outputs
    if task == 'lits':
        _, pred_fp = torch.max(out, 1)  # NDHW
        assert pred_fp.dim() == 4
        nClass = 3
        nums = get_class_num_lits(pred_fp, body_mask)  # number of voxels of each class
    elif task == 'brats':
        pred_fp = (torch.sigmoid(out) >= 0.5).int()  # N3DHW, 3 for WT, TC, ET
        nClass = 4  # bkg, WT, TC, ET
        nums = get_class_num_brats(pred_fp, body_mask)
    else:
        raise RuntimeError(f'Unknown task {task}')

    weight_map = {}
    if 'p:' in style:
        p = float(style[2:])
        for i in range(nClass):
            if nums[i] == 0:
                weight_map[i] = 1.0
            else:
                weight_map[i] = (1 / nums[i] * max(nums)) ** p
    else:
        raise RuntimeError(f'Unknown attention weight map style {style}')
    return weight_map, nums


def tune_activation_range(model, output_fp, data_batch, max_iter=1000, need_init=False):
    """
    1) initilize alpha_act by mse
    2) train alpha_act end-to-end

    :param max_iter:
    :param need_init:
    :param model:
    :param output_fp:
    :param data_batch:
    :return:
    """
    if need_init:
        set_init_alpha(model)
        model(data_batch)
    set_quantized(model)

    opt_param = []
    for module in model.modules():
        if isinstance(module, PTQConv):
            opt_param.append(module.alpha_act)
    optimizer = optim.Adam(opt_param, lr=5e-4)

    loss_all = []
    for i in range(max_iter):
        out_q = model(data_batch)
        loss = F.mse_loss(out_q, output_fp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())

    return loss_all


def plot_save(losses, file: str):
    plt.figure()
    plt.plot(losses)
    plt.savefig(file)
    plt.close()


def do_ptq(args, model_cube, data_cube, tester: PTQTester, snap_dir):
    model = model_cube['model']
    pretrain = model_cube['pretrain']
    device = torch.device(args.device)

    # -- load weights
    print('pretrain is :', pretrain)
    sd = torch.load(pretrain, map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=False)
    model.eval()
    # fold bn
    search_fold_and_remove_bn(model)
    model.to(device)

    # -- prepare calibration data
    data_batch, label_batch = get_calibration_data(args, data_cube)
    data_batch = data_batch.to(device)
    if args.lwq_verbose:
        # check data shape
        print(f'data_batch shape {data_batch.shape}, label_batch shape {label_batch.shape}')
        print('Calibration data shape:', data_batch.shape)

    set_name(model)
    set_snapdir(model, snap_dir)
    set_fp(model)

    # -- 1st test (FP)
    if args.test_fp:
        tester.test_as_is(folder='fp', is_save_nii=args.save_nii)

    # register hooks
    hook_handles = []

    def register_hook_recursive(module: nn.Module, handles: list):
        if isinstance(module, (PTQConv, PTQBlock)):
            h1 = module.register_forward_hook(forward_hook)
            # h2 = module.register_full_backward_hook(backward_hook)
            handles.append(h1)
            # handles.append(h2)
            # prune if cur node is PTQConv or PTQBlock (i.e., no more branching)
        else:
            for m in module.children():
                register_hook_recursive(m, handles)
        # return handles

    register_hook_recursive(model, hook_handles)

    # infer once
    model.to(device)
    data_batch = data_batch.to(device)
    model.eval()
    t0 = time.time()
    set_fp(model)
    output_fp = model(data_batch).detach()

    if args.task == 'brats':
        body_mask = (data_batch[:,0] != 0.0).bool()  # NDHW
    else:
        body_mask = torch.ones_like(data_batch[:,0]).bool()  # NDHW
    print(f'Body occupies {body_mask.sum() / body_mask.numel() * 100}% of the volume.')
    weight_map, nums = get_att_weight_map(output_fp, torch.ones_like(data_batch[:,0]).bool(), 'p:0.5',
                                          task=args.task)
    mask_pyramid = get_mask_pyramid(output_fp, body_mask, weight_map, args.init_stride, num_lvls=5,
                                    task=args.task)
    set_mask(model, mask_pyramid)  # pass mask pyramid to every PTQConv

    with open(P.join(snap_dir, 'class_voxel_nums.txt'), 'w') as fid:
        for n in nums:
            fid.write(f'{n}\n')

    # remove handles
    for h in hook_handles:
        h.remove()

    layer_loss = []
    set_anything(model, 'layer_loss', layer_loss)
    # -- PTQ
    t1 = time.time()
    set_quantizing(model)
    with torch.no_grad():
        output_q = model(data_batch)
    t2 = time.time()
    set_quantized(model)
    # print time cost
    print(f'FP forward costs {t1 - t0:.3f}s, PTQ costs {t2 - t1:.3f}s, totally {t2 - t0:.3f}s.')
    with open(P.join(snap_dir, 'time_cost.txt'), 'w') as fid:
        fid.write(f'{(t2 - t0)/60:.3f} min.')
    with open(P.join(snap_dir, 'layer_loss.txt'), 'w') as fid:
        fid.write('\n'.join(layer_loss))

    niis_q = extract_nii(output_q, task=args.task)
    niis_fp = extract_nii(output_fp, task=args.task)
    print(output_fp.shape)
    for i in range(output_fp.shape[1]):
        niis_q[i].to_filename(P.join(snap_dir, f'Qseg{i}.nii.gz'))
        niis_fp[i].to_filename(P.join(snap_dir, f'FPseg{i}.nii.gz'))

    if not args.no_test:
        tester.test_as_is('ptq', args.save_nii)

    # save quantized model
    model.cpu()
    tester.snapshot('state_in_fp.pkl', compress=False)
    store_int_weight(model)
    tester.snapshot('state_in_int8.pkl', compress=False)
    tester.snapshot('state_in_int8_compress.npz', compress=True)
