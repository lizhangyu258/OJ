import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _mark_dynamic(tensor, dims):
    for dim in dims:
        torch._dynamo.mark_dynamic(tensor, dim)


class DynamicGatedSequenceReduction(torch.nn.Module):
    """动态 batch/sequence 的 gate + broadcast + sequence reduction。"""

    def forward(self, x, residual, gate, bias):
        x_fp32 = x.to(torch.float32)
        residual_fp32 = residual.to(torch.float32)
        gate_fp32 = gate.to(torch.float32)
        bias_fp32 = bias.to(torch.float32)
        gated = (x_fp32 + residual_fp32 * 0.25 + bias_fp32) * torch.sigmoid(gate_fp32)
        activated = torch.relu(gated) + x_fp32 * 0.125
        seq_mean = activated.mean(dim=1, keepdim=True)
        centered = activated - seq_mean
        seq_rms = torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + 1e-4)
        return centered * seq_rms * 0.75 + activated * 0.25


def build_testcase():
    device = 'npu'
    model = DynamicGatedSequenceReduction().to(device)
    x = torch.randn((16, 128, 512), requires_grad=False, dtype=torch.float16, device=device) * 0.25
    residual = torch.randn((16, 128, 512), requires_grad=False, dtype=torch.float16, device=device) * 0.25
    gate = torch.randn((16, 128, 1), requires_grad=False, dtype=torch.float16, device=device) * 0.25
    bias = torch.randn((1, 1, 512), requires_grad=False, dtype=torch.float16, device=device) * 0.125
    _mark_dynamic(x, (0, 1))
    _mark_dynamic(residual, (0, 1))
    _mark_dynamic(gate, (0, 1))
    return {
        "model_or_func": model,
        "inputs": (x, residual, gate, bias),
        "device": device,
        "compile_options": {"dynamic": True},
    }
