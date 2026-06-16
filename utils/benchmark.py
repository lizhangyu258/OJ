import os
import logging
import torch
import torch_npu
from contextlib import contextmanager
from torch._inductor.utils import run_and_get_code
from typing import Any, Callable, Optional, Tuple

from utils.profiler import setup_profiler_output, find_and_parse_step_trace, get_default_prof_dir

logger = logging.getLogger(__name__)


def _resolve_tool_bin_dir(bin_config: Optional[dict], key: str) -> Optional[str]:
    if not bin_config:
        return None

    bin_dir = bin_config.get(key)
    if bin_dir is None:
        return None
    if not isinstance(bin_dir, str) or not bin_dir.strip():
        raise ValueError(f"bin_config[{key!r}] must be a non-empty string")

    resolved_bin_dir = os.path.abspath(bin_dir.strip())
    if not os.path.isdir(resolved_bin_dir):
        raise FileNotFoundError(f"Configured bin directory does not exist: {resolved_bin_dir}")

    for tool_name in ("bishengir-compile", "bishengir-opt"):
        tool_path = os.path.join(resolved_bin_dir, tool_name)
        if not os.path.isfile(tool_path):
            raise FileNotFoundError(f"Required tool not found: {tool_path}")

    return resolved_bin_dir


@contextmanager
def _patched_tool_bin_dir(bin_dir: Optional[str]):
    if not bin_dir:
        yield
        return

    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{original_path}" if original_path else bin_dir
    logger.info("Using bishengir tools from: %s", bin_dir)
    try:
        yield
    finally:
        os.environ["PATH"] = original_path


def _reset_compile_state():
    reset_func = getattr(getattr(torch, "_dynamo", None), "reset", None)
    if callable(reset_func):
        reset_func()


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
    compile_options: Optional[dict],
    tool_bin_dir: Optional[str] = None,
):
    """
    公共辅助函数：准备模型并编译
    """
    if isinstance(model_or_func, torch.nn.Module):
        model = model_or_func
        model.to(device)
    else:
        model = model_or_func

    _reset_compile_state()
    with _patched_tool_bin_dir(tool_bin_dir):
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
    artifact_subdir: Optional[str] = None,
    run_name: str = ""
) -> Tuple[Optional[str], Optional[float]]:
    if prof_config is None:
        prof_config = get_default_prof_config()
    
    all_step_num = warmup_steps + exec_steps
    prof_output_dir = setup_profiler_output(get_default_prof_dir(), artifact_subdir)
    
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
    warmup_steps: int = 10,
    exec_steps: int = 30,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    prof_config=None,
    artifact_subdir: Optional[str] = None,
    bin_config: Optional[dict] = None,
) -> dict:
    setup_environment()
    
    if compile_options is None:
        compile_options = {"dynamic": False}
    
    if prof_config is None:
        prof_config = get_default_prof_config()
    
    baseline_bin_dir = _resolve_tool_bin_dir(bin_config, "baseline")
    current_bin_dir = _resolve_tool_bin_dir(bin_config, "current")

    model, baseline_compile_func, _, _ = _prepare_model_and_compile(
        model_or_func,
        inputs,
        device,
        compile_options,
        tool_bin_dir=baseline_bin_dir,
    )
    _, current_compile_func, compile_out, codes = _prepare_model_and_compile(
        model_or_func,
        inputs,
        device,
        compile_options,
        tool_bin_dir=current_bin_dir,
    )

    results = {
        "compile_out": compile_out,
        "codes": codes,
    }
    
    if isinstance(model, torch.nn.Module):
        eager_result = model(*inputs)
    else:
        eager_result = model(*inputs)
    graph_result = current_compile_func(*inputs)
    
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
        artifact_subdir,
        "eager"
    )
    results["eager_time"] = eager_time
    if eager_time is not None:
        logger.info(f"[eager] Average execution time: {eager_time} us")
    
    _, compile_time = run_profiler(
        baseline_compile_func,
        inputs,
        warmup_steps,
        exec_steps,
        prof_config,
        artifact_subdir,
        "compile"
    )
    results["compile_time"] = compile_time
    if compile_time is not None:
        logger.info(f"[compile] Average execution time: {compile_time} us")
    
    _, current_time = run_profiler(
        current_compile_func,
        inputs,
        warmup_steps,
        exec_steps,
        prof_config,
        artifact_subdir,
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
