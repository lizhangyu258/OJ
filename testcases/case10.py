import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


class FusedAttentionScore(torch.nn.Module):
    """使用 fused attention 原语，避免手写分解 softmax 链路。"""

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, q, k, v, bias):
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        fused = context + q * 0.1
        fused = fused * 0.75 + torch.relu(context) * 0.25
        return (fused + fused.mean(dim=-1, keepdim=True) * 0.05).to(torch.float32)


def build_testcase():
    device = 'npu'
    d_k = 64
    model = FusedAttentionScore(d_k=d_k).to(device)
    q = torch.randn(4, 8, 128, d_k, dtype=torch.float16, device=device)
    k = torch.randn(4, 8, 128, d_k, dtype=torch.float16, device=device)
    v = torch.randn(4, 8, 128, d_k, dtype=torch.float16, device=device)
    bias = torch.randn(1, 1, 128, 128, dtype=torch.float16, device=device)
    return {
        "model_or_func": model,
        "inputs": (q, k, v, bias),
        "device": device,
    }
