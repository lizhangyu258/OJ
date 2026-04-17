import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fused_add_mul_with_cast(x, y, z, bias):
    """带broadcast、cast和归约的点乘融合算子。"""
    mixed = (x + y) * z + bias
    mixed = mixed.to(torch.float16)
    stabilized = mixed + mixed.mean(dim=-1, keepdim=True) * 0.125
    return stabilized.to(torch.float32)


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((1, 2048), requires_grad=False, dtype=torch.float32, device=device)
    z = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    bias = torch.randn((256, 1), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": fused_add_mul_with_cast,
        "inputs": (x, y, z, bias),
        "device": device,
    }
