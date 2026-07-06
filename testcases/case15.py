import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


class TelemetryBalanceGateFusion(torch.nn.Module):
    """Blend a baseline-corrected telemetry path with a channel gate."""

    def forward(self, readings, channel_profile):
        balance_adjustment = readings - readings.mean(dim=-1, keepdim=True)
        balance_adjustment = balance_adjustment.sum(dim=-1, keepdim=True)
        balance_adjustment = balance_adjustment * 0.125 + torch.tanh(balance_adjustment) * 0.03125

        telemetry_gate = torch.sigmoid(channel_profile).reshape(1, 1, -1)
        telemetry_gate = telemetry_gate * 0.75 + torch.tanh(telemetry_gate * 0.5) * 0.125

        mixed = balance_adjustment + telemetry_gate
        return mixed * 0.875 + torch.tanh(mixed) * 0.125


def build_testcase():
    device = 'npu'
    model = TelemetryBalanceGateFusion().to(device)
    readings = torch.randn((32, 128, 256), requires_grad=False, dtype=torch.float32, device=device) * 0.125
    channel_profile = torch.randn((256,), requires_grad=False, dtype=torch.float32, device=device) * 0.125

    return {
        "model_or_func": model,
        "inputs": (readings, channel_profile),
        "device": device,
        "compile_options": {"dynamic": False},
        "rtol": 1e-4,
        "atol": 1e-4,
    }
