import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 环境变量调用需要在torch_npu初始化之前
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
from utils.profiler import setup_profiler_output, find_and_parse_step_trace, get_default_prof_dir
from utils import setup_logging

# config导入在compile执行之前
torch._inductor.config.npu_backend = "mlir"

class TestCase1(torch.nn.Module):
    def forward(self, x, y):
        return (x * y) + (x - y)


def main():
    setup_logging()
    
    prof_config = torch_npu.profiler._ExperimentalConfig(
        export_type = [torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
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
    print("codes[0]: ", codes[0])

    eager_result = model(x, y)
    graph_result = compiled_model(x, y)
    if torch.npu.is_available():
        torch.npu.synchronize()

    # 直接比较整个张量而不是逐行比较
    if torch.allclose(eager_result, graph_result, rtol=1e-5, atol=1e-5):
        print("Precision test result: Passed")
    else:
        print("Precision test result: Failed")
        print(f"Expected: {eager_result}")
        print(f"Actual: {graph_result}")
        print(f"max diff: ", (eager_result - graph_result).abs().max().item())

    warmup_stem_num = 5
    exec_step_num = 10
    all_step_num = warmup_stem_num + exec_step_num
    prof_output_dir = setup_profiler_output(get_default_prof_dir())

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
        experimental_config=prof_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_output_dir)
    )

    prof.start()
    with torch.no_grad():
        for _ in range(all_step_num):
            outputs = compiled_model(x, y)
            if torch.npu.is_available():
                torch.npu.synchronize()
            prof.step()
    prof.stop()

    print(f"\nProfiler results saved to: {prof_output_dir}")
    avg_exec_time = find_and_parse_step_trace(prof_output_dir)
    if avg_exec_time is not None:
        print(f"Average execution time: {avg_exec_time} us")


if __name__ == "__main__":
    main()
