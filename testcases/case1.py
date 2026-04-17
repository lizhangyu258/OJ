import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def fused_ops(x, y):
    """融合多个元素级操作：(x + y) * (x - y) + x"""
    return (x + y) * (x - y) + x


def build_testcase():
    device = 'npu'
    x = torch.randn((64, 128), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((64, 128), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": fused_ops,
        "inputs": (x, y),
        "device": device,
    }
