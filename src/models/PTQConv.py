"""
@author: rongzhao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_helper import *


class PTQConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,  # basic convolution parameters
                 q_weight=True, qlvl=8, q_act=True, qlvl_act=8,  # basic quantization parameters
                 **kwQ):  # auxiliary quantization parameters / config
        super(PTQConv, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias)
        self.conv_param = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.q_act = q_act
        self.q_weight = q_weight
        self.qlvl_w = qlvl
        self.qlvl_act = qlvl_act
        self.kwQ = kwQ

        # quantization range parameters
        self.alpha_act = nn.Parameter(torch.tensor(1.))
        self.alpha_w = nn.Parameter(torch.tensor(1.))

        # necessary intermediate variables
        self.act_in = None  # the original FP input
        self.output_fp = None  # the original FP output (serve as the target)
        self.grad_in = None  # the grad wrt the input
        self.grad_out = None  # the grad wrt the output

        # information for intermediate outputs
        self.name = None
        self.snap_dir = kwQ.get('snap_dir', None)

        # FP weights backup
        self.w_backup = None
        self.b_backup = None

        # forward flags
        self._fp = True
        self._quantizing = False
        self._quantized = False
        self._init_act = False
        self._act_inited = False

    def set_fp(self):
        self._fp = True
        self._quantizing = False
        self._quantized = False
        self._init_act = False

    def set_quantizing(self):
        self._fp = False
        self._quantizing = True
        self._quantized = False
        self._init_act = False

    def set_quantized(self):
        self._fp = False
        self._quantizing = False
        self._quantized = True
        self._init_act = False

    def set_init_act(self):
        self._fp = False
        self._quantizing = False
        self._quantized = False
        self._init_act = True

    def init_alpha_act(self, x):
        a, b = project_by_iter(x, self.qlvl_act, 0, 1)
        self.alpha_act.data = torch.tensor(a, device=x.device)
        self._act_inited = True
        return a * b.detach()

    def qweight_init_iter(self):
        pass

    def qparam_init(self):
        pass

    def perform_quantization(self):
        pass

    def backup_weight(self):
        self.w_backup = self.weight.data.cpu().clone()
        if self.bias is not None:
            self.b_backup = self.bias.data.cpu().clone()

    '''
    method of discretization:
    1. symmetric [-a, a]  # -> IN USE <- # 
        delta = 2*a / (num_lvl - 1)
        qvar = clip(var / a, -1, 1)
        round((qvar + a) / delta) => [0, num_lvl-1] (integers)
        integer * delta - a => qvar
    2. asymmetric [-b, a]
        delta = (a+b) / (num_lvl - 1)
        zero_point = b / delta
        qvar = clip(var, -b, a)
        round((qvar + b) / delta) => [0, num_lvl-1] (integers)
        integer * delta - b => qvar
    3. non-negative [0, a]
    '''

    def _quantize_w(self):
        qweight = discretize(self.weight / self.alpha_w, self.qlvl_w, -1, 1) * self.alpha_w
        return qweight

    def _quantize_act(self, x):
        qact = discretize(x / self.alpha_act, self.qlvl_act, 0, 1) * self.alpha_act
        return qact

    def ptq(self, x):
        """
        After PTQ, qweight is saved in self.weight, self.alpha_act and self.alpha_w
        contain the quantization ranges for activation and weights, respectively.
        """
        raise NotImplementedError

    def store_int_weight(self):
        """
        Convert self.weight to UINT8 or INT32
        For storage only
        Remember to restore it to FP for inference
        """
        # suppose q = a * b
        a = self.alpha_w.data
        q = self.weight.data
        b = q / a  # b \in [-1, 1]
        delta = 2 / (self.qlvl_w - 1)
        w_int = torch.round((b + 1) / delta)
        if self.qlvl_w <= 256:
            w_int = w_int.to(torch.uint8)
        else:
            w_int = w_int.to(torch.int32)
        self.weight.requires_grad = False
        self.weight.data = w_int.data.cpu()

    def restore_fp_weight(self):
        """
        Restore INT weights to FP (discrete in value, FP in type)
        """
        w_int = self.weight.data
        delta = 2 / (self.qlvl_w - 1)
        b = w_int.float() * delta - 1
        q = self.alpha_w.data * b
        self.weight.data = q

    def forward(self, x):
        if self._fp:
            return F.conv3d(x, self.weight, self.bias, **self.conv_param)
        elif self._quantizing:
            self.ptq(x)
            qact = self._quantize_act(x) if self.q_act else x  # quantize input by its range alpha_act
            qweight = self.weight  # assume the quantized weights have been stored
            qbias = self.bias  # no need for quantization
            return F.conv3d(qact, qweight, qbias, **self.conv_param)
        elif self._quantized:
            qact = self._quantize_act(x) if self.q_act else x  # quantize input by its range alpha_act
            qweight = self.weight  # assume the quantized weights have been stored
            qbias = self.bias  # no need for quantization
            return F.conv3d(qact, qweight, qbias, **self.conv_param)
        elif self._init_act:
            qact = self.init_alpha_act(x)
            return F.conv3d(qact, self.weight, self.bias, **self.conv_param)
        else:
            # raise config error
            raise RuntimeError(f"Unknown FP/Quant setting: FP={self._fp}, "
                               f"Quantizing={self._quantizing}, Quantized={self._quantized}")



