import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fused_ops_with_broadcast_and_reduce(x, y, bias, scale):
    """带cast、broadcast和归约的融合链路。"""
    x_half = x.to(torch.float16)
    y_half = y.to(torch.float16)
    mixed = (x_half * y_half) + (x_half - y_half)
    fused = (mixed + bias) * scale
    fused = fused + (x_half * 0.25) - (y_half * 0.125)
    fused_fp32 = fused.to(torch.float32)
    row_mean = fused_fp32.mean(dim=-1, keepdim=True)
    return fused_fp32 + row_mean * 0.1


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    bias = torch.randn((1, 2048), requires_grad=False, dtype=torch.float16, device=device)
    scale = torch.randn((256, 1), requires_grad=False, dtype=torch.float16, device=device)
    return {
        "model_or_func": fused_ops_with_broadcast_and_reduce,
        "inputs": (x, y, bias, scale),
        "device": device,
    }
