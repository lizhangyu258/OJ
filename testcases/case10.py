import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


class FusedAttentionScore(torch.nn.Module):
    """融合的注意力分数计算：Q*K^T / sqrt(d_k) + softmax"""
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / (d_k ** 0.5)
    
    def forward(self, q, k):
        scores = q @ k.transpose(-2, -1) * self.scale
        return scores.softmax(dim=-1)


def main():
    setup_logging()
    
    device = 'npu'
    d_k = 64
    model = FusedAttentionScore(d_k=d_k).to(device)
    artifact_subdir = os.path.splitext(os.path.basename(__file__))[0]
    
    q = torch.randn(8, 12, d_k, device=device)
    k = torch.randn(8, 12, d_k, device=device)
    
    results = benchmark(
        model_or_func=model,
        inputs=(q, k),
        device=device,
        warmup_steps=5,
        exec_steps=10,
        artifact_subdir=artifact_subdir
    )
    return results


if __name__ == "__main__":
    main()
