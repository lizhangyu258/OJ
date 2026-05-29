import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def complex_activations(x, y, bias):
    """单 dtype 的激活融合链路。"""
    x_fp32 = x.to(torch.float32)
    y_fp32 = y.to(torch.float32)
    bias_fp32 = bias.to(torch.float32)
    shifted = x_fp32 + bias_fp32
    sig = torch.sigmoid(shifted)
    tanh_val = torch.tanh(y_fp32 - bias_fp32)
    relu_val = torch.relu(x_fp32 + y_fp32 + bias_fp32)
    mixed = sig * tanh_val + relu_val * 0.5
    fused = mixed - sig * relu_val * 0.25 + x_fp32 * 0.125
    channel_mean = fused.mean(dim=-1, keepdim=True)
    return fused + channel_mean * 0.2


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device) * 0.5
    y = torch.randn((256, 2048), requires_grad=False, dtype=torch.float16, device=device) * 0.5
    bias = torch.randn((1, 2048), requires_grad=False, dtype=torch.float16, device=device) * 0.5
    return {
        "model_or_func": complex_activations,
        "inputs": (x, y, bias),
        "device": device,
    }
