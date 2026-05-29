import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _mark_dynamic(tensor, dims):
    for dim in dims:
        torch._dynamo.mark_dynamic(tensor, dim)


class DynamicLayerNormChain(torch.nn.Module):
    """动态 batch/sequence 的归一化链路。"""

    def forward(self, x, residual, weight, bias):
        x_fp32 = x.to(torch.float32) + residual.to(torch.float32) * 0.25
        mean = x_fp32.mean(dim=-1, keepdim=True)
        centered = x_fp32 - mean
        var = centered.square().mean(dim=-1, keepdim=True)
        normed = centered * torch.rsqrt(var + 1e-4)
        out = normed * weight.to(torch.float32) + bias.to(torch.float32)
        return out + x_fp32 * 0.05


def build_testcase():
    device = 'npu'
    model = DynamicLayerNormChain().to(device)
    x = torch.randn((32, 160, 512), requires_grad=False, dtype=torch.float16, device=device) * 0.25
    residual = torch.randn((32, 160, 512), requires_grad=False, dtype=torch.float16, device=device) * 0.25
    weight = torch.randn((512,), requires_grad=False, dtype=torch.float16, device=device) * 0.125 + 1.0
    bias = torch.randn((512,), requires_grad=False, dtype=torch.float16, device=device) * 0.125
    _mark_dynamic(x, (0, 1))
    _mark_dynamic(residual, (0, 1))
    return {
        "model_or_func": model,
        "inputs": (x, residual, weight, bias),
        "device": device,
        "compile_options": {"dynamic": True},
    }
