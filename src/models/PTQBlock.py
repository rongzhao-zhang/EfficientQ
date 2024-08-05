import torch.nn as nn
from .factory_blk import ResBlockWithType, ReLU
from .PTQConv import PTQConv


class PTQBlock(ResBlockWithType):
    def __init__(self, inChans, outChans, drop_rate=0.5, dilation=1, nla=ReLU(True),
                 Conv=PTQConv, bn=nn.BatchNorm3d, blk_type='pre'):
        super().__init__(inChans, outChans, drop_rate, dilation, nla, Conv, bn, blk_type)
        assert isinstance(self.block1.conv, PTQConv)
        self.q_act = self.block1.conv.q_act
        self.q_weight = self.block1.conv.q_weight
        self.qlvl_w = self.block1.conv.qlvl
        self.qlvl_act = self.block1.conv.qlvl_act
        self.channel_wise = self.block1.conv.channel_wise
        self.kwQ = self.block1.conv.kwQ

        # necessary intermediate variables
        self.input_fp = None  # the original FP input
        self.output_fp = None  # the original FP output (serve as the target)
        self.grad_in = None  # the grad wrt the input
        self.grad_out = None  # the grad wrt the output

        # information for intermediate outputs
        self.name = None
        self.snap_dir = self.kwQ.get('snap_dir', None)

        # forward flags
        self._fp = True
        self._quantizing = False
        self._quantized = False

    def set_fp(self):
        self._fp = True
        self._quantizing = False
        self._quantized = False

    def set_quantizing(self):
        self._fp = False
        self._quantizing = True
        self._quantized = False

    def set_quantized(self):
        self._fp = False
        self._quantizing = False
        self._quantized = True

    def ptq(self, x):
        raise NotImplementedError

    def forward(self, x):
        if self._fp:
            self.block1.conv.set_fp()
            self.block2.conv.set_fp()
            return super().forward(x)
        elif self._quantizing:
            self.ptq(x)
            self.block1.conv.set_quantized()
            self.block2.conv.set_quantized()
            return super().forward(x)
        elif self._quantized:
            self.block1.conv.set_quantized()
            self.block2.conv.set_quantized()
            return super().forward(x)
        else:
            # raise config error
            raise RuntimeError(f"Unknown FP/Quant setting: FP={self._fp}, "
                               f"Quantizing={self._quantizing}, Quantized={self._quantized}")


