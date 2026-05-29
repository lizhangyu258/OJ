import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _mark_dynamic(tensor, dims):
    for dim in dims:
        torch._dynamo.mark_dynamic(tensor, dim)


def dynamic_broadcast_reduce(x, bias, scale):
    """动态 batch/hidden 的 broadcast + elementwise + row reduction。"""
    x_fp32 = x.to(torch.float32)
    bias_fp32 = bias.to(torch.float32)
    scale_fp32 = scale.to(torch.float32)
    fused = (x_fp32 + bias_fp32) * scale_fp32
    fused = torch.relu(fused) + x_fp32 * 0.125
    row_mean = fused.mean(dim=-1, keepdim=True)
    centered = fused - row_mean
    return centered * 0.75 + fused * 0.25


def build_testcase():
    device = 'npu'
    x = torch.randn((192, 1024), requires_grad=False, dtype=torch.float16, device=device)
    bias = torch.randn((1, 1024), requires_grad=False, dtype=torch.float16, device=device)
    scale = torch.randn((192, 1), requires_grad=False, dtype=torch.float16, device=device)
    _mark_dynamic(x, (0, 1))
    _mark_dynamic(bias, (1,))
    _mark_dynamic(scale, (0,))
    return {
        "model_or_func": dynamic_broadcast_reduce,
        "inputs": (x, bias, scale),
        "device": device,
        "compile_options": {"dynamic": True},
    }
