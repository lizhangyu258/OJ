import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class VectorOps(torch.nn.Module):
    """组合向量操作：乘法、加法、减法"""
    def forward(self, x, y):
        return (x * y) + (x - y)


def build_testcase():
    model = VectorOps()
    device = 'npu'
    x = torch.randn(128, 128, device=device)
    y = torch.randn(128, 128, device=device)
    return {
        "model_or_func": model,
        "inputs": (x, y),
        "device": device,
    }
