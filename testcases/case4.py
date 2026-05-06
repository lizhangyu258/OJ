import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fused_add_mul_with_cast(x, y, z, bias):
    """单 dtype 的点乘 epilogue，更适合作为 compile 融合样例。"""
    mixed = (x + y) * z + bias
    stabilized = torch.relu(mixed) + mixed * 0.25
    return (stabilized + stabilized.mean(dim=-1, keepdim=True) * 0.125).to(torch.float32)


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device)
    y = torch.randn((1, 2048), requires_grad=False, dtype=torch.float16, device=device)
    z = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device)
    bias = torch.randn((256, 1), requires_grad=False, dtype=torch.float16, device=device)
    return {
        "model_or_func": fused_add_mul_with_cast,
        "inputs": (x, y, z, bias),
        "device": device,
    }
