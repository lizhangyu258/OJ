import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class LayerNorm(torch.nn.Module):
    """简单的LayerNorm实现"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        x = (x - mean) * (var + self.eps).rsqrt()
        return x * self.weight + self.bias


def build_testcase():
    device = 'npu'
    model = LayerNorm(hidden_size=512).to(device)
    x = torch.randn(32, 512, device=device)
    return {
        "model_or_func": model,
        "inputs": (x,),
        "device": device,
    }
