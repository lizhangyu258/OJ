import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class MatMulWithFusion(torch.nn.Module):
    """更适合compile融合的matmul后处理链路。"""

    def forward(self, x, y, row_bias, col_scale, residual):
        x_fp32 = x.to(torch.float32)
        y_fp32 = y.to(torch.float32)
        residual_fp32 = residual.to(torch.float32)
        matmul_result = x_fp32 @ y_fp32
        fused = torch.relu(matmul_result + row_bias.to(torch.float32))
        fused = fused * col_scale.to(torch.float32) + residual_fp32
        row_mean = fused.mean(dim=-1, keepdim=True)
        return fused + row_mean * 0.05 + residual_fp32 * 0.02


def build_testcase():
    model = MatMulWithFusion()
    device = 'npu'
    x = torch.randn(128, 512, dtype=torch.float16, device=device) * 0.0625
    y = torch.randn(512, 512, dtype=torch.float16, device=device) * 0.0625
    row_bias = torch.randn(1, 512, dtype=torch.float16, device=device) * 0.03125 + 0.25
    col_scale = torch.randn(128, 1, dtype=torch.float16, device=device) * 0.125
    residual = torch.randn(128, 512, dtype=torch.float16, device=device) * 0.0625
    return {
        "model_or_func": model,
        "inputs": (x, y, row_bias, col_scale, residual),
        "device": device,
    }
