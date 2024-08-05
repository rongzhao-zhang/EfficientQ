import argparse
import yaml
from train_seg import train_fp
from ptq_seg import ptq


class CFG(object):
    def __init__(self):
        self.attr_list = []

    def append(self, **kwargs):
        for name in kwargs:
            self.attr_list.append(name)
            setattr(self, name, kwargs[name])


def merge_config(cfg: str, args: argparse.Namespace):
    """Configuration file first, if not specified in config, then use commandline args
    In other words, use specified in config to replace input arguments
    """
    with open(cfg, 'r') as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)

    for k, v in config.items():
        if v is not None:
            setattr(args, k, v)

    return args


# -- command line arguments -- #
# general settings
parser = argparse.ArgumentParser(description='Entrance for Quantization/FP training/Inference')
parser.add_argument('mission', choices=['train_fp', 'ptq'])
parser.add_argument('--pretrain')
parser.add_argument('--resume')
parser.add_argument('--device', default=0, type=int, dest='device',
                    help='GPU ID.')
parser.add_argument('--task')
parser.add_argument('--suffix', default="", type=str, dest='suffix', help='folder name suffix.')
parser.add_argument('--test_fp', action='store_true')

# read configuration
parser.add_argument('--config', type=str)

# data config
parser.add_argument('--data_dir')
parser.add_argument('--split_dir')
parser.add_argument('--round', default='1', type=str, dest='round',
                    help='round number.')
parser.add_argument('--patch_size')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--test_batch_size', default=1, type=int)
parser.add_argument('--crop_type', default='random')
parser.add_argument('--balance_rate', type=float)
parser.add_argument('--data_on_disk', action='store_true')
parser.add_argument('--bin_label', help='convert to binary label')
parser.add_argument('--multi_label', help='multiple labels per pixel')
parser.add_argument('--merge_type', help='how to merge multiple labels')
parser.add_argument('--random_noise_p', type=float)
parser.add_argument('--access_type', default='npy')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--da_scaling', type=str, default=None)
parser.add_argument('--scal_order', type=int, default=1)

# model config
parser.add_argument('--model', default='UResQ')
parser.add_argument('--nMod', type=int)
parser.add_argument('--nClass', type=int)
parser.add_argument('--init_stride', type=str, default='1')
parser.add_argument('--resblock')
parser.add_argument('--depth')
parser.add_argument('--width')
parser.add_argument('--dilation')
parser.add_argument('--nla', default='relu')
parser.add_argument('--norm', type=str, default='bn')
parser.add_argument('--group_num', type=int, help='GN\'s group number')
parser.add_argument('--drop_rate', default=0.2, type=float)
parser.add_argument('--no_drop', action='store_true')
parser.add_argument('--ds', type=str, default=None, choices=['simple', 'complex', ''])
parser.add_argument('--init_kernel', default=3, type=int)
parser.add_argument('--block_type', default='RBpre')
parser.add_argument('--hetero_dim', action='store_true')
parser.add_argument('--blk', type=str, default='pre')

# FP training config
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--max_epoch', type=int, default=20)
parser.add_argument('--loss', type=str, default='CE')
parser.add_argument('--test_interval', type=int, default=50)
parser.add_argument('--disp_interval', type=int, default=10)
parser.add_argument('--weight_decay', type=str, default='0')
parser.add_argument('--no_test', action='store_true')
parser.add_argument('--exp_id', type=str, default=None)

# quantization config
parser.add_argument('--qconv', default='conv')
parser.add_argument('--qlvl_w', type=int)
parser.add_argument('--qlvl_a', type=int)
parser.add_argument('--q_first', help='whether quantize first layer. e.g., --q_first 256,64 for W8A4')
parser.add_argument('--q_last', help='similar to q_first')

# PTQ config
parser.add_argument('--debug', action='store_true')

parser.add_argument('--lwq_dataid', type=int, default=0)
parser.add_argument('--lwq_batchsz', type=int, default=1)
parser.add_argument('--lwq_patchsz')

parser.add_argument('--lwq_verbose', action='store_true')

# evaluation config
parser.add_argument('--save_nii', action='store_true')

args = parser.parse_args()

if args.config:
    new_args = merge_config(args.config, args)
else:
    new_args = args

if new_args.mission == 'train_fp':
    train_fp(new_args)
elif new_args.mission == 'ptq':
    ptq(new_args)
else:
    raise NotImplementedError



