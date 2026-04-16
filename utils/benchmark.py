import os
import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
from typing import Any, Callable, Optional, Tuple

from utils.profiler import setup_profiler_output, find_and_parse_op_statistic, get_default_prof_dir


def setup_environment():
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'
    torch._inductor.config.npu_backend = "mlir"


def get_default_prof_config():
    return torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
    )


def precision_check(eager_result, graph_result, rtol=1e-5, atol=1e-5) -> Tuple[bool, Optional[float]]:
    if torch.npu.is_available():
        torch.npu.synchronize()
    
    passed = torch.allclose(eager_result, graph_result, rtol=rtol, atol=atol)
    max_diff = None
    
    if not passed:
        max_diff = (eager_result - graph_result).abs().max().item()
    
    return passed, max_diff


def run_benchmark(
    model_or_func: Any,
    inputs: Tuple[Any, ...],
    device: str = 'npu',
    compile_options: Optional[dict] = None,
    warmup_steps: int = 5,
    exec_steps: int = 10,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    prof_config=None,
    skip_profiler: bool = False,
    skip_precision: bool = False
) -> dict:
    setup_environment()
    
    if compile_options is None:
        compile_options = {"dynamic": False}
    
    if prof_config is None:
        prof_config = get_default_prof_config()
    
    if isinstance(model_or_func, torch.nn.Module):
        model = model_or_func
        model.to(device)
    else:
        model = model_or_func
    
    compile_func = torch.compile(model, **compile_options)
    compile_out, codes = run_and_get_code(compile_func, *inputs)
    print("compile_out: ", compile_out)
    print("codes[0]: ", codes[0])
    
    results = {
        "compile_out": compile_out,
        "codes": codes,
    }
    
    if not skip_precision:
        if isinstance(model, torch.nn.Module):
            eager_result = model(*inputs)
        else:
            eager_result = model(*inputs)
        graph_result = compile_func(*inputs)
        
        passed, max_diff = precision_check(eager_result, graph_result, rtol=rtol, atol=atol)
        if passed:
            print("Precision test result: Passed")
        else:
            print("Precision test result: Failed")
            print(f"Expected: {eager_result}")
            print(f"Actual: {graph_result}")
            if max_diff is not None:
                print(f"max diff: {max_diff}")
        
        results["precision_passed"] = passed
        results["eager_result"] = eager_result
        results["graph_result"] = graph_result
        results["max_diff"] = max_diff
    
    if not skip_profiler:
        all_step_num = warmup_steps + exec_steps
        prof_output_dir = setup_profiler_output(get_default_prof_dir())
        
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup_steps, active=exec_steps, repeat=1),
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
                outputs = compile_func(*inputs)
                if torch.npu.is_available():
                    torch.npu.synchronize()
                prof.step()
        prof.stop()
        
        print(f"\nProfiler results saved to: {prof_output_dir}")
        avg_exec_time = find_and_parse_op_statistic(prof_output_dir)
        if avg_exec_time is not None:
            print(f"Average execution time: {avg_exec_time} us")
        
        results["prof_output_dir"] = prof_output_dir
        results["avg_exec_time"] = avg_exec_time
    
    return results
