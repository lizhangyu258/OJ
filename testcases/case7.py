import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def vector_reduction(x):
    """共享统计量的归一化链路，减少独立 reduction 次数。"""
    mean_val = x.mean(dim=-1, keepdim=True)
    centered = x - mean_val
    var = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(var + 1e-4)
    fused = normalized * 0.75 + x * 0.125
    return fused + fused.mean(dim=-1, keepdim=True) * 0.05


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": vector_reduction,
        "inputs": (x,),
        "device": device,
    }
