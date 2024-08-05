import torch
from utils.tester import Tester, PTQTester
import definer
from ptqer import do_ptq


def ptq(args):
    # -- manual argument parse -- #
    device = torch.device(args.device)
    # data loader
    data_cube, data_info, nMod, nClass, patch_size = definer.get_data_cube(args)
    # convolution type
    QConv, Qinfo, kwQ = definer.get_conv_class(args)
    # -- model definition -- #
    model_cube, model_info = definer.get_model_cube(args, QConv, kwQ)
    model = model_cube['model']

    # load weights
    sd = torch.load(args.pretrain, map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=False)
    model.to(device)

    # -- snapshot dir
    snapshot_scheme = definer.get_snapshot_config(args, model_info, Qinfo, model, data_cube)
    snapshot_root = snapshot_scheme['root']

    # -- Tester definition --
    tester = PTQTester(model_cube, data_cube, snapshot_scheme, device)

    do_ptq(args, model_cube, data_cube, tester, snapshot_root)

    # -- the end -- #
