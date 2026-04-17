import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import run_full_benchmark, setup_logging


class TestCase5(torch.nn.Module):
    def forward(self, x, y):
        return (x * y) + (x - y)


def main():
    setup_logging()
    
    model = TestCase5()
    device = 'npu'
    
    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    
    results = run_full_benchmark(
        model_or_func=model,
        inputs=(x, y),
        device=device,
        warmup_steps=5,
        exec_steps=10
    )


if __name__ == "__main__":
    main()
