import os
import logging
import torch
import torch_npu
from torch._inductor.utils import run_and_get_code
from typing import Any, Callable, Optional, Tuple

from utils.profiler import setup_profiler_output, find_and_parse_step_trace, get_default_prof_dir

logger = logging.getLogger(__name__)


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


def _prepare_model_and_compile(
    model_or_func: Any,
    inputs: Tuple[Any, ...],
    device: str,
    compile_options: Optional[dict]
):
    """
    公共辅助函数：准备模型并编译
    """
    if isinstance(model_or_func, torch.nn.Module):
        model = model_or_func
        model.to(device)
    else:
        model = model_or_func
    
    compile_func = torch.compile(model, **compile_options)
    compile_out, codes = run_and_get_code(compile_func, *inputs)
    logger.info(f"compile_out: {compile_out}")
    logger.info(f"codes[0]: {codes[0]}")
    
    return model, compile_func, compile_out, codes


def _log_precision_result(passed, eager_result, graph_result, max_diff):
    """
    公共辅助函数：记录精度检查结果
    """
    if passed:
        logger.info("Precision test result: Passed")
    else:
        logger.error("Precision test result: Failed")
        logger.error(f"Expected: {eager_result}")
        logger.error(f"Actual: {graph_result}")
        if max_diff is not None:
            logger.error(f"max diff: {max_diff}")


def run_profiler(
    func: Callable,
    inputs: Tuple[Any, ...],
    warmup_steps: int,
    exec_steps: int,
    prof_config=None,
    run_name: str = ""
) -> Tuple[Optional[str], Optional[float]]:
    if prof_config is None:
        prof_config = get_default_prof_config()
    
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
            outputs = func(*inputs)
            if torch.npu.is_available():
                torch.npu.synchronize()
            prof.step()
    prof.stop()
    
    run_name_str = f" ({run_name})" if run_name else ""
    logger.info(f"\nProfiler results{run_name_str} saved to: {prof_output_dir}")
    avg_exec_time = find_and_parse_step_trace(prof_output_dir)
    return prof_output_dir, avg_exec_time


def benchmark(
    model_or_func: Any,
    inputs: Tuple[Any, ...],
    device: str = 'npu',
    compile_options: Optional[dict] = None,
    warmup_steps: int = 5,
    exec_steps: int = 10,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    prof_config=None
) -> dict:
    setup_environment()
    
    if compile_options is None:
        compile_options = {"dynamic": False}
    
    if prof_config is None:
        prof_config = get_default_prof_config()
    
    model, compile_func, compile_out, codes = _prepare_model_and_compile(
        model_or_func, inputs, device, compile_options
    )
    
    results = {
        "compile_out": compile_out,
        "codes": codes,
    }
    
    if isinstance(model, torch.nn.Module):
        eager_result = model(*inputs)
    else:
        eager_result = model(*inputs)
    graph_result = compile_func(*inputs)
    
    passed, max_diff = precision_check(eager_result, graph_result, rtol=rtol, atol=atol)
    _log_precision_result(passed, eager_result, graph_result, max_diff)
    
    results["precision_passed"] = passed
    results["eager_result"] = eager_result
    results["graph_result"] = graph_result
    results["max_diff"] = max_diff
    
    _, eager_time = run_profiler(
        model if isinstance(model, torch.nn.Module) else model_or_func,
        inputs,
        warmup_steps,
        exec_steps,
        prof_config,
        "eager"
    )
    results["eager_time"] = eager_time
    if eager_time is not None:
        logger.info(f"[eager] Average execution time: {eager_time} us")
    
    # TODO: 实现compile_time与current_time的程序来源切换逻辑
    # 当前 compile_time 应使用 CANN 包标准安装目录下的 bishengir-compile/bishengir-opt 被测程序
    _, compile_time = run_profiler(
        compile_func,
        inputs,
        warmup_steps,
        exec_steps,
        prof_config,
        "compile"
    )
    results["compile_time"] = compile_time
    if compile_time is not None:
        logger.info(f"[compile] Average execution time: {compile_time} us")
    
    # TODO: 实现compile_time与current_time的程序来源切换逻辑
    # 当前 current_time 应使用用户上传的被测程序
    _, current_time = run_profiler(
        compile_func,
        inputs,
        warmup_steps,
        exec_steps,
        prof_config,
        "current"
    )
    results["current_time"] = current_time
    if passed and eager_time is not None and compile_time is not None and compile_time > 0 and eager_time > 0:
        speedup = eager_time / compile_time
        logger.info(f"Speedup: {speedup:.4f}x")
        results["speedup"] = speedup
    
    logger.info("\n=== Benchmark Summary ===")
    logger.info(f"Precision test: {'Passed' if passed else 'Failed'}")
    if eager_time is not None:
        logger.info(f"[eager] Average execution time: {eager_time} us")
    if compile_time is not None:
        logger.info(f"[compile] Average execution time: {compile_time} us")
    if current_time is not None:
        logger.info(f"[current] Average execution time: {current_time} us")
    logger.info("========================")
    
    return results
