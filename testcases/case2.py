import os
# 环境变量调用需要在torch_npu初始化之前
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
import torch

# config导入在compile执行之前
torch._inductor.config.npu_backend = "mlir"

class TestCase1(torch.nn.Module):
    def forward(self, x, y):
        return (x * y) + (x - y)

prof_config = torch_npu.profiler._ExperimentalConfig(
    export_type = [torch_npu.profiler.ExportType.Text],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

warmup_stem_num = 5
exec_step_num = 10
all_step_num = warmup_stem_num + exec_step_num
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
prof_output_dir = os.path.join(ROOT_DIR, "prof")
os.makedirs(prof_output_dir, exist_ok=True)

prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup_stem_num, active=exec_step_num, repeat=1),
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=prof_config
)

model = TestCase1()
device = 'npu'
model.to(device)

# 定义输入张量
x = torch.randn(10, 10, device=device)
y = torch.randn(10, 10, device=device)

compiled_model = torch.compile(model, dynamic=False)
compile_out, codes = run_and_get_code(compiled_model,x,y)
print("compile_out: ", compile_out)
print("codes: ", codes)

prof.start()
with torch.no_grad():
    for _ in range(all_step_num):
        outputs = compiled_model(x, y)
        if torch.npu.is_available():
            torch.npu.synchronize()
        prof.step()
prof.stop()