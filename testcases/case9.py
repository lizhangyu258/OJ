import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def complex_activations(x, y, bias):
    """单 dtype 的激活融合链路。"""
    shifted = x + bias
    sig = torch.sigmoid(shifted)
    tanh_val = torch.tanh(y - bias)
    relu_val = torch.relu(x + y + bias)
    mixed = sig * tanh_val + relu_val * 0.5
    fused = mixed - sig * relu_val * 0.25 + x * 0.125
    channel_mean = fused.mean(dim=-1, keepdim=True)
    return (fused + channel_mean * 0.2).to(torch.float32)


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device)
    y = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device)
    bias = torch.randn((1, 2048), requires_grad=False, dtype=torch.float16, device=device)
    return {
        "model_or_func": complex_activations,
        "inputs": (x, y, bias),
        "device": device,
    }
