import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class VectorOps(torch.nn.Module):
    """共享统计量的向量融合链路，避免多次独立 reduction。"""

    def forward(self, x, y, gate):
        mixed = (x * y) + x
        fused = mixed * gate + y * 0.25
        row_mean = fused.mean(dim=-1, keepdim=True)
        centered = fused - row_mean
        inv_rms = (centered.square().mean(dim=-1, keepdim=True) + 1e-4).rsqrt()
        return (centered * inv_rms + fused * 0.05).to(torch.float32)


def build_testcase():
    model = VectorOps()
    device = 'npu'
    x = torch.randn(256, 2048, dtype=torch.float16, device=device)
    y = torch.randn(256, 2048, dtype=torch.float16, device=device)
    gate = torch.randn(1, 2048, dtype=torch.float16, device=device)
    return {
        "model_or_func": model,
        "inputs": (x, y, gate),
        "device": device,
    }
