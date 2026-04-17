import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


def fused_add_mul(x, y, z):
    """融合加法和乘法操作"""
    return x + y * z


def main():
    setup_logging()
    
    device = 'npu'
    artifact_subdir = os.path.splitext(os.path.basename(__file__))[0]
    x = torch.randn((256, 256), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((256, 256), requires_grad=False, dtype=torch.float32, device=device)
    z = torch.randn((256, 256), requires_grad=False, dtype=torch.float32, device=device)
    
    results = benchmark(
        model_or_func=fused_add_mul,
        inputs=(x, y, z),
        device=device,
        warmup_steps=5,
        exec_steps=10,
        artifact_subdir=artifact_subdir
    )
    return results


if __name__ == "__main__":
    main()
