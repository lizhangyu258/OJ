import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class VectorOps(torch.nn.Module):
    """带broadcast、cast和双归约的向量融合算子。"""

    def forward(self, x, y, gate):
        x_half = x.to(torch.float16)
        y_half = y.to(torch.float16)
        gate_half = gate.to(torch.float16)
        mixed = (x_half * y_half) + (x_half - y_half)
        fused = mixed * gate_half + (x_half * 0.5)
        fused_fp32 = fused.to(torch.float32)
        row_sum = fused_fp32.sum(dim=-1, keepdim=True)
        row_mean = fused_fp32.mean(dim=-1, keepdim=True)
        return fused_fp32 + row_sum * 0.001 + row_mean * 0.1


def build_testcase():
    model = VectorOps()
    device = 'npu'
    x = torch.randn(256, 2048, device=device)
    y = torch.randn(256, 2048, device=device)
    gate = torch.randn(1, 2048, device=device)
    return {
        "model_or_func": model,
        "inputs": (x, y, gate),
        "device": device,
    }
