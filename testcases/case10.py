import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class FusedAttentionScore(torch.nn.Module):
    """融合的注意力分数计算：Q*K^T / sqrt(d_k) + softmax"""
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / (d_k ** 0.5)
    
    def forward(self, q, k):
        scores = q @ k.transpose(-2, -1) * self.scale
        return scores.softmax(dim=-1)


def build_testcase():
    device = 'npu'
    d_k = 64
    model = FusedAttentionScore(d_k=d_k).to(device)
    q = torch.randn(8, 12, d_k, device=device)
    k = torch.randn(8, 12, d_k, device=device)
    return {
        "model_or_func": model,
        "inputs": (q, k),
        "device": device,
    }
