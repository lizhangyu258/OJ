import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class VectorOps(torch.nn.Module):
    """共享统计量的向量融合链路，避免多次独立 reduction。"""

    def forward(self, x, y, gate):
        x_fp32 = x.to(torch.float32)
        y_fp32 = y.to(torch.float32)
        gate_fp32 = gate.to(torch.float32)
        mixed = (x_fp32 * y_fp32) + x_fp32
        fused = mixed * gate_fp32 + y_fp32 * 0.25
        row_mean = fused.mean(dim=-1, keepdim=True)
        centered = fused - row_mean
        inv_rms = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + 1e-3)
        return centered * inv_rms + fused * 0.05


def build_testcase():
    model = VectorOps()
    device = 'npu'
    x = torch.randn(256, 2048, dtype=torch.float16, device=device) * 0.25
    y = torch.randn(256, 2048, dtype=torch.float16, device=device) * 0.25
    gate = torch.randn(1, 2048, dtype=torch.float16, device=device) * 0.25
    return {
        "model_or_func": model,
        "inputs": (x, y, gate),
        "device": device,
    }
