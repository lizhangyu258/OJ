import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import run_benchmark, setup_logging


def op_calc(x, y):
    return x * y


def main():
    setup_logging()
    
    device = 'npu'
    x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device=device)
    y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device=device)
    
    results = run_benchmark(
        model_or_func=op_calc,
        inputs=(x, y),
        device=device,
        compile_options={"options": {"npu_backend": "mlir"}},
        warmup_steps=5,
        exec_steps=10
    )


if __name__ == "__main__":
    main()
