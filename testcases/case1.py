import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fused_ops_with_broadcast_and_reduce(x, y, bias, scale):
    """单 dtype 的 broadcast + epilogue + 归约，更利于 compile 融合。"""
    mixed = (x * y) + (x - y)
    fused = (mixed + bias) * scale
    fused = fused + torch.relu(x + bias) * 0.125 - y * 0.0625
    row_mean = fused.mean(dim=-1, keepdim=True)
    centered = fused - row_mean
    return (centered * 0.9 + fused * 0.1).to(torch.float32)


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device)
    y = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device)
    bias = torch.randn((1, 2048), requires_grad=False, dtype=torch.float16, device=device)
    scale = torch.randn((256, 1), requires_grad=False, dtype=torch.float16, device=device)
    return {
        "model_or_func": fused_ops_with_broadcast_and_reduce,
        "inputs": (x, y, bias, scale),
        "device": device,
    }
