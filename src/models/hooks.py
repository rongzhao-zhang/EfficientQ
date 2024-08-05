import torch
import torch.nn as nn


def forward_hook(m: nn.Module, i: torch.Tensor, o: torch.Tensor):
    m.output_fp = o.detach().cpu()


def backward_hook(m: nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor):
    m.grad_out = grad_out[0].detach().cpu()

