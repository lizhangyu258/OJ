import logging
import os
import shutil
import uuid
import torch
import torch_npu
from contextlib import contextmanager, nullcontext
from torch._inductor.utils import run_and_get_code
from typing import Any, Callable, Dict, Optional, Tuple

from utils.profiler import find_and_parse_step_trace
from utils.profiler import get_default_prof_dir
from utils.profiler import get_default_traced_graph_cache_dir
from utils.profiler import resolve_output_dir
from utils.profiler import setup_profiler_output
from utils.binary_manager import tool_binary_context
from utils.tool_validation import validate_tool_bin_dir

logger = logging.getLogger(__name__)

CACHE_ENV_NAMES = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_COMPILE_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TRACED_GRAPH_CACHE_DIR",
    "BISHENGIR_CACHE_DIR",
    "MLIR_CACHE_DIR",
    "XDG_CACHE_HOME",
)


def _resolve_tool_bin_dir(bin_config: Optional[dict], key: str) -> Optional[str]:
    if not bin_config:
        return None

    bin_dir = bin_config.get(key)
    if bin_dir is None:
        return None
    if not isinstance(bin_dir, str) or not bin_dir.strip():
        raise ValueError(f"bin_config[{key!r}] must be a non-empty string")

    return validate_tool_bin_dir(bin_dir.strip(), key)


@contextmanager
def _patched_environment(updates: Dict[str, str]):
    original_values = {}
    missing_keys = set()
    for key, value in updates.items():
        if key in os.environ:
            original_values[key] = os.environ[key]
        else:
            missing_keys.add(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key in updates:
            if key in missing_keys:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_values[key]


def _cleanup_compile_cache(cache_dir: str):
    if os.path.exists(cache_dir):
        if not os.path.isdir(cache_dir):
            raise NotADirectoryError(f"Compile cache path is not a directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)


def _make_phase_cache_dir(artifact_subdir: Optional[str], phase: str) -> str:
    cache_root = resolve_output_dir(get_default_traced_graph_cache_dir(), artifact_subdir)
    return os.path.join(cache_root, f"{phase}-{uuid.uuid4().hex}")


@contextmanager
def _compile_cache_environment(cache_dir: str, *, clean: bool = True):
    resolved_cache_dir = os.path.abspath(cache_dir)
    if clean:
        _cleanup_compile_cache(resolved_cache_dir)
    else:
        os.makedirs(resolved_cache_dir, exist_ok=True)

    updates = {name: os.path.join(resolved_cache_dir, name.lower()) for name in CACHE_ENV_NAMES}
    for value in updates.values():
        os.makedirs(value, exist_ok=True)

    with _patched_environment(updates):
        yield


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
    cache_dir: Optional[str] = None,
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
    cache_context = _compile_cache_environment(cache_dir) if cache_dir else nullcontext()
    with cache_context, tool_binary_context(tool_bin_dir):
        torch._dynamo.reset()
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
    run_name: str = "",
    tool_bin_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[float]]:
    if prof_config is None:
        prof_config = get_default_prof_config()
    
    all_step_num = warmup_steps + exec_steps
    run_name_str = f" ({run_name})" if run_name else ""
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
    try:
        cache_context = _compile_cache_environment(cache_dir, clean=False) if cache_dir else nullcontext()
        with torch.no_grad(), cache_context, tool_binary_context(tool_bin_dir):
            for _ in range(all_step_num):
                outputs = func(*inputs)
                if torch.npu.is_available():
                    torch.npu.synchronize()
                prof.step()
    finally:
        prof.stop()

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
    baseline_cache_dir = _make_phase_cache_dir(artifact_subdir, "baseline")
    current_cache_dir = _make_phase_cache_dir(artifact_subdir, "current")

    model, baseline_compile_func, _, _ = _prepare_model_and_compile(
        model_or_func,
        inputs,
        device,
        compile_options,
        tool_bin_dir=baseline_bin_dir,
        cache_dir=baseline_cache_dir,
    )
    _, current_compile_func, compile_out, codes = _prepare_model_and_compile(
        model_or_func,
        inputs,
        device,
        compile_options,
        tool_bin_dir=current_bin_dir,
        cache_dir=current_cache_dir,
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
        "eager",
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
        "compile",
        tool_bin_dir=baseline_bin_dir,
        cache_dir=baseline_cache_dir,
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
        "current",
        tool_bin_dir=current_bin_dir,
        cache_dir=current_cache_dir,
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
