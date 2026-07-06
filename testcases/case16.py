import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class RetailFulfillmentScorer(torch.nn.Module):
    """Blend shelf telemetry, promotion vectors, and sample windows."""

    def forward(self, live_signal, seasonal_shift, sample_windows, sku_affinity, aisle_bias, window_weight):
        sku_gate = torch.sigmoid(sku_affinity).reshape(1, -1, 1)
        sku_gain = (torch.tanh(sku_affinity) * 0.125 + 1.0).reshape(1, -1, 1)
        regional_bias = aisle_bias.reshape(1, 1, -1)

        shelf_signal = live_signal.permute(0, 2, 1)
        shelf_shift = seasonal_shift.permute(0, 2, 1)
        demand_score = torch.exp(shelf_signal * sku_gain + shelf_shift * sku_gate + regional_bias)

        sample_kernel = torch.sigmoid(window_weight).reshape(1, 1, 1, -1)
        window_samples = sample_windows.permute(0, 2, 1, 3)
        window_score = (window_samples * sample_kernel).sum(dim=-1).to(torch.float32)

        demand_score = torch.relu(demand_score + window_score)
        demand_score = demand_score * (1.0 + sku_gate.to(torch.float32) * 0.015625)
        return demand_score.permute(0, 2, 1)


def build_testcase():
    device = 'npu'
    model = RetailFulfillmentScorer().to(device)
    live_signal = torch.randn((32, 128, 64), requires_grad=False, dtype=torch.float32, device=device) * 0.125
    seasonal_shift = torch.randn((32, 128, 64), requires_grad=False, dtype=torch.float32, device=device) * 0.125
    sample_windows = torch.randn((32, 128, 64, 16), requires_grad=False, dtype=torch.float32, device=device) * 0.125
    sku_affinity = torch.randn((64,), requires_grad=False, dtype=torch.float32, device=device) * 0.125
    aisle_bias = torch.randn((128,), requires_grad=False, dtype=torch.float32, device=device) * 0.03125
    window_weight = torch.randn((16,), requires_grad=False, dtype=torch.float32, device=device) * 0.125
    return {
        "model_or_func": model,
        "inputs": (live_signal, seasonal_shift, sample_windows, sku_affinity, aisle_bias, window_weight),
        "device": device,
        "compile_options": {"dynamic": False},
    }
