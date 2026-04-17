import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class MatMulWithBias(torch.nn.Module):
    """带偏置的矩阵乘法"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_dim, out_dim))
        self.bias = torch.nn.Parameter(torch.randn(out_dim))
    
    def forward(self, x):
        return x @ self.weight + self.bias


def build_testcase():
    device = 'npu'
    model = MatMulWithBias(in_dim=256, out_dim=128).to(device)
    x = torch.randn(64, 256, device=device)
    return {
        "model_or_func": model,
        "inputs": (x,),
        "device": device,
    }
