import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class MatMulWithFusion(torch.nn.Module):
    """更适合compile融合的matmul后处理链路。"""

    def forward(self, x, y, row_bias, col_scale, residual):
        matmul_result = x @ y
        fused = torch.relu(matmul_result + row_bias)
        fused = fused * col_scale + residual
        fused = fused.to(torch.float16).to(torch.float32)
        row_mean = fused.mean(dim=-1, keepdim=True)
        col_mean = fused.mean(dim=0, keepdim=True)
        return fused + row_mean * 0.05 + col_mean * 0.02


def build_testcase():
    model = MatMulWithFusion()
    device = 'npu'
    x = torch.randn(128, 512, device=device)
    y = torch.randn(512, 512, device=device)
    row_bias = torch.randn(1, 512, device=device)
    col_scale = torch.randn(128, 1, device=device)
    residual = torch.randn(128, 512, device=device)
    return {
        "model_or_func": model,
        "inputs": (x, y, row_bias, col_scale, residual),
        "device": device,
    }
