import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _mark_dynamic(tensor, dims):
    for dim in dims:
        torch._dynamo.mark_dynamic(tensor, dim)


class DynamicMatMulEpilogue(torch.nn.Module):
    """动态 M/K/N 的 matmul + broadcast epilogue。"""

    def forward(self, x, weight, row_bias, col_bias):
        x_fp32 = x.to(torch.float32)
        weight_fp32 = weight.to(torch.float32)
        out = x_fp32 @ weight_fp32
        out = torch.relu(out + row_bias.to(torch.float32))
        out = out + col_bias.to(torch.float32)
        return out + out.mean(dim=-1, keepdim=True) * 0.05


def build_testcase():
    device = 'npu'
    model = DynamicMatMulEpilogue().to(device)
    x = torch.randn((96, 384), requires_grad=False, dtype=torch.float16, device=device) * 0.125
    weight = torch.randn((384, 256), requires_grad=False, dtype=torch.float16, device=device) * 0.125
    row_bias = torch.randn((1, 256), requires_grad=False, dtype=torch.float16, device=device) * 0.125
    col_bias = torch.randn((96, 1), requires_grad=False, dtype=torch.float16, device=device) * 0.125
    _mark_dynamic(x, (0, 1))
    _mark_dynamic(weight, (0, 1))
    _mark_dynamic(row_bias, (1,))
    _mark_dynamic(col_bias, (0,))
    return {
        "model_or_func": model,
        "inputs": (x, weight, row_bias, col_bias),
        "device": device,
        "compile_options": {"dynamic": True},
    }
