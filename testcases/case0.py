import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class FusedBroadcastReduction(torch.nn.Module):
    """带广播与归约的多段融合链路，适合作为高收益基线用例。"""

    def forward(self, x, y):
        z = (x * y) + (x - y)
        w = (x * x) + (y * 0.5) + 1.0
        res = (z * w) - ((z + w) * 0.1)
        res = res + (y * 1.5)
        row_sum = torch.sum(res, dim=1)
        row_mean = torch.mean(res, dim=1)
        out = torch.mean(row_sum + row_mean)
        return out


def build_testcase():
    device = "npu"
    model = FusedBroadcastReduction().to(device)
    x = torch.randn(16, 1000, device=device)
    y = torch.randn(16, 1000, device=device)
    return {
        "model_or_func": model,
        "inputs": (x, y),
        "device": device,
    }
