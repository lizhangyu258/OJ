import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def vector_reduction(x):
    """归约与broadcast回写结合，提升融合收益。"""
    sum_val = x.sum(dim=-1, keepdim=True)
    mean_val = x.mean(dim=-1, keepdim=True)
    sq_mean = (x * x).mean(dim=-1, keepdim=True)
    normalized = (x - mean_val) * torch.rsqrt(sq_mean + 1e-4)
    fused = normalized + sum_val * 0.0005 + x * 0.125
    return fused.sum(dim=-1)


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": vector_reduction,
        "inputs": (x,),
        "device": device,
    }
