import os
# 环境变量调用需要在torch_npu初始化之前
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
import torch_npu._inductor
import torch
import triton

# config导入在compile执行之前
torch._inductor.config.npu_backend = "mlir"

# 定义模型
def op_calc (x , y):
    return x * y
x = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
y = torch.randn((3,), requires_grad=False, dtype=torch.float32, device="npu")
std_out = op_calc(x, y)

# options调用，修改compile参数
compile_func = torch.compile(op_calc, options={"npu_backend": "mlir"})
compile_out, codes = run_and_get_code(compile_func,x,y)
print(codes[0])