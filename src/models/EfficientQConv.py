"""
@author: rongzhao
"""
import torch
import torch.nn.functional as F
from .PTQConv import PTQConv
from .layer_helper import *
import time
from . import solver


class EfficientQConv(PTQConv):
    """
    EfficientQConv with analytical proximal iter
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,  # basic convolution parameters
                 q_weight=True, qlvl=8, q_act=True, qlvl_act=8,  # basic quantization parameters
                 **kwQ):  # auxiliary quantization parameters / config
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         q_weight, qlvl, q_act, qlvl_act, **kwQ)
        # for post-training optimization
        self.lwq_iter = 200
        self.lwq_rho = 10
        self.lwq_rho_max = 1000
        self.lwq_eta = 1
        self.lwq_fold_bn = True
        self.lwq_verbose = kwQ.get('lwq_verbose', False)

        self.mask_pyramid = None
        self.layer_loss = None

    def ptq(self, x):
        """
        The core function of post-training quantization.
        """

        device = x.device
        G = self.weight.data  # a * b.detach()
        dual = torch.zeros_like(self.weight)
        out_fp = self.output_fp.detach().to(device)
        x = x.detach()

        admm_iter = self.lwq_iter
        y_dim = out_fp.numel()
        y_std = out_fp.std().item()
        w_dim = G.numel()
        w_std = G.std().item()
        rho_scale = max(y_dim*y_std / (w_dim*w_std), 1.0)

        att = None
        print(f'Calibrating {self.name}')
        if self.mask_pyramid:
            for mask in self.mask_pyramid:
                # print(f'mask.shape {mask.shape} v.s. x.shape {x.shape}')
                if mask.shape[1:] == out_fp.shape[2:]:
                    att = mask
                    # print(f'mask shape is {mask.shape}, mean is {att.mean().item()}')
                    break
        if att is not None:
            rho_scale *= att.mean().item()
            att = att.to(device)

        if self.q_act:
            if self._act_inited:
                Qactivation = self._quantize_act(x)
            else:
                a_act, b_act = project_by_iter(x, self.qlvl_act, 0, 1)
                self.alpha_act.data = torch.tensor(a_act, dtype=x.dtype, device=x.device)
                Qactivation = a_act * b_act.detach()  # self._quantize_act(x)
        else:
            Qactivation = x

        rho = self.lwq_rho * rho_scale
        rho_m = self.lwq_rho_max * rho_scale
        eta = self.lwq_eta * rho_scale

        _, _, kD, kH, kW = self.weight.shape
        a_history = []
        loss_history = []
        t0 = time.time()
        W0 = self.weight.data.clone()
        if self.bias is not None:
            b0 = self.bias.data.clone()
        else:
            b0 = None

        qsolver = solver.QuadraSolver(Qactivation.detach(), out_fp.detach(), kD, kH, kW,
                                      self.stride, self.padding,
                                      device=device, mu=0, eta=eta,
                                      W0=W0, att=att, b0=b0)
        b_star = self.bias
        bestG = None
        bestB = None
        bestLoss = 1e10
        pres = 0  # primal_residual
        dres = 0  # dual_residual
        i = 0
        while i < admm_iter:
            # -- Proximal step -- #
            if self.bias is not None:
                w_star, b_star = qsolver.solve(rho, eta, G - dual)
            else:
                w_star = qsolver.solve(rho, eta, G - dual)

            # projection step
            G0 = G
            a_w, b_w = project_by_iter(w_star + dual, self.qlvl_w, -1, 1)
            G = a_w * b_w
            # dual update
            dual = w_star - G + dual

            # compute residuals
            if self.lwq_verbose:
                pres = (w_star - G).norm().item()
                dres = rho * (G-G0).norm().item()

            with torch.no_grad():
                out_q = F.conv3d(Qactivation, G.float().to(x.device), b_star, self.stride,
                                 self.padding, self.dilation, self.groups)
                lossf = F.mse_loss(out_q, out_fp).item()
                loss_history.append(lossf)

            if i % 10 == 0 and self.lwq_verbose:  # print every 10 admm iters
                print(f'ADMM iter {i+1}: primal residual = {pres:.4f}, '
                      f'dual residual = {dres:.4f}, rho = {rho:.4f}, eta = {eta:.4f}, '
                      f'loss = {lossf:.7f}.')

            N = 50  # change rho per N iters
            if i % N == 0:
                if rho*2 <= rho_m:
                    rho *= 2
                    dual /= 2
                    # dual.zero_()
                else:
                    dual /= (rho_m / rho)
                    rho = rho_m

            if i == 0 or lossf < bestLoss:
                bestG = G
                bestB = b_star if self.bias is not None else None
                bestLoss = lossf

            i += 1

        # save quantized weight
        G = bestG

        if bestG is not None:
            G = bestG
        if bestB is not None:
            B = bestB
        # G = G.float().to(x.device)
        # store optimal values
        self.weight.data = G
        if self.bias is not None:
            self.bias.data = B
        self.alpha_w.data = torch.tensor(a_w, dtype=x.dtype, device=x.device)

        # final test
        out_q = F.conv3d(Qactivation, G, self.bias, self.stride, self.padding,
                         self.dilation, self.groups)
        lossf = F.mse_loss(out_q, out_fp).item()
        if att is not None:
            lossf = (att.unsqueeze(dim=1) * ((out_q - out_fp) ** 2)).mean().item()
        self.layer_loss.append(f'{self.name:45s}:{lossf}')

    def compute_quant_error(self, output_fp, Qw, Qact):
        with torch.no_grad():
            output_q = F.conv3d(Qact, Qw, self.bias, **self.conv_param)
            err = F.mse_loss(output_fp, output_q)
        return err.item()


