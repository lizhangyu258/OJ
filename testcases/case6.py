import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


class RMSNorm(torch.nn.Module):
    """简单的RMSNorm实现"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x / (norm + self.eps)
        return x * self.weight


def main():
    setup_logging()
    
    device = 'npu'
    model = RMSNorm(hidden_size=512).to(device)
    artifact_subdir = os.path.splitext(os.path.basename(__file__))[0]
    
    x = torch.randn(32, 512, device=device)
    
    results = benchmark(
        model_or_func=model,
        inputs=(x,),
        device=device,
        warmup_steps=5,
        exec_steps=10,
        artifact_subdir=artifact_subdir
    )


if __name__ == "__main__":
    main()
