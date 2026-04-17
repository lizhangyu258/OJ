import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def complex_activations(x, y, bias):
    """激活、cast、broadcast和归约组合。"""
    x_half = x.to(torch.float16)
    y_half = y.to(torch.float16)
    bias_half = bias.to(torch.float16)
    sig = torch.sigmoid(x_half)
    tanh_val = torch.tanh(y_half)
    relu_val = torch.relu(x_half + y_half + bias_half)
    mixed = sig * tanh_val + relu_val * 0.5 - sig * relu_val
    mixed_fp32 = mixed.to(torch.float32)
    channel_mean = mixed_fp32.mean(dim=-1, keepdim=True)
    return mixed_fp32 + channel_mean * 0.2


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((256, 2048), requires_grad=False, dtype=torch.float32, device=device)
    bias = torch.randn((1, 2048), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": complex_activations,
        "inputs": (x, y, bias),
        "device": device,
    }
