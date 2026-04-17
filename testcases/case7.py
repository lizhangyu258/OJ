import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import benchmark, setup_logging


def vector_reduction(x):
    """向量归约操作：求和、最大值、平均值"""
    sum_val = x.sum(dim=-1)
    max_val = x.max(dim=-1)[0]
    mean_val = x.mean(dim=-1)
    return sum_val + max_val - mean_val


def main():
    setup_logging()
    
    device = 'npu'
    artifact_subdir = os.path.splitext(os.path.basename(__file__))[0]
    x = torch.randn((128, 512), requires_grad=False, dtype=torch.float32, device=device)
    
    results = benchmark(
        model_or_func=vector_reduction,
        inputs=(x,),
        device=device,
        warmup_steps=5,
        exec_steps=10,
        artifact_subdir=artifact_subdir
    )
    return results


if __name__ == "__main__":
    main()
