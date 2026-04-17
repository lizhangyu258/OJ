import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class FusedAttentionScore(torch.nn.Module):
    """更大shape的attention链路，附带bias与后处理。"""

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / (d_k ** 0.5)
    
    def forward(self, q, k, v, bias):
        scores = q @ k.transpose(-2, -1) * self.scale
        scores = scores + bias
        probs = scores.softmax(dim=-1)
        context = probs @ v
        return context + q * 0.1


def build_testcase():
    device = 'npu'
    d_k = 64
    model = FusedAttentionScore(d_k=d_k).to(device)
    q = torch.randn(4, 8, 128, d_k, device=device)
    k = torch.randn(4, 8, 128, d_k, device=device)
    v = torch.randn(4, 8, 128, d_k, device=device)
    bias = torch.randn(1, 1, 128, 128, device=device)
    return {
        "model_or_func": model,
        "inputs": (q, k, v, bias),
        "device": device,
    }
