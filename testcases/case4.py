import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fused_add_mul(x, y, z):
    """融合加法和乘法操作"""
    return x + y * z


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 256), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((256, 256), requires_grad=False, dtype=torch.float32, device=device)
    z = torch.randn((256, 256), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": fused_add_mul,
        "inputs": (x, y, z),
        "device": device,
    }
