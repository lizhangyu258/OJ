import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


class MatMulWithFusion(torch.nn.Module):
    """矩阵乘法与后续操作的融合：(x @ y) * z + w"""
    def forward(self, x, y, z, w):
        matmul_result = x @ y
        fused_result = matmul_result * z + w
        return fused_result


def main():
    setup_logging()
    
    model = MatMulWithFusion()
    device = 'npu'
    
    x = torch.randn(64, 128, device=device)
    y = torch.randn(128, 64, device=device)
    z = torch.randn(64, 64, device=device)
    w = torch.randn(64, 64, device=device)
    
    results = benchmark(
        model_or_func=model,
        inputs=(x, y, z, w),
        device=device,
        warmup_steps=5,
        exec_steps=10
    )


if __name__ == "__main__":
    main()
