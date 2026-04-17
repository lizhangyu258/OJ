import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


class VectorOps(torch.nn.Module):
    """组合向量操作：乘法、加法、减法"""
    def forward(self, x, y):
        return (x * y) + (x - y)


def main():
    setup_logging()
    
    model = VectorOps()
    device = 'npu'
    artifact_subdir = os.path.splitext(os.path.basename(__file__))[0]
    
    x = torch.randn(128, 128, device=device)
    y = torch.randn(128, 128, device=device)
    
    results = benchmark(
        model_or_func=model,
        inputs=(x, y),
        device=device,
        warmup_steps=5,
        exec_steps=10,
        artifact_subdir=artifact_subdir
    )


if __name__ == "__main__":
    main()
