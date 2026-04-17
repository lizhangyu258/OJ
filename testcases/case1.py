import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


def fused_ops(x, y):
    """融合多个元素级操作：(x + y) * (x - y) + x"""
    return (x + y) * (x - y) + x


def main():
    setup_logging()
    
    device = 'npu'
    x = torch.randn((64, 128), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((64, 128), requires_grad=False, dtype=torch.float32, device=device)
    
    results = benchmark(
        model_or_func=fused_ops,
        inputs=(x, y),
        device=device,
        warmup_steps=5,
        exec_steps=10
    )


if __name__ == "__main__":
    main()
