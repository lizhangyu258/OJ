import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def complex_activations(x, y):
    """复杂的激活函数组合，避免单op fallback"""
    sig = torch.sigmoid(x)
    tanh_val = torch.tanh(y)
    relu_val = torch.relu(x + y)
    return sig * tanh_val + relu_val * 0.5 - sig * relu_val


def build_testcase():
    device = 'npu'
    x = torch.randn((256, 512), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((256, 512), requires_grad=False, dtype=torch.float32, device=device)
    return {
        "model_or_func": complex_activations,
        "inputs": (x, y),
        "device": device,
    }
