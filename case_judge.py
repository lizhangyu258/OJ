import argparse
import os
import json
import glob
import importlib.util
import logging
import signal
from contextlib import contextmanager

from utils.judge import build_empty_result
from utils.judge import generate_final_result
from utils.judge import load_baseline_data

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 测试用例目录
TESTCASES_DIR = os.path.join(ROOT_DIR, 'testcases')
# Baseline数据目录
BASELINE_DIR = os.path.join(ROOT_DIR, 'baseline')
BASELINE_DATA_FILE = os.path.join(BASELINE_DIR, 'data.yaml')
TESTCASE_TIMEOUT_SECONDS = 300


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run testcase evaluation.")
    parser.add_argument(
        "--clean-up",
        action="store_true",
        help="Remove generated artifact directories after evaluation finishes.",
    )
    return parser.parse_args(argv)

# 获取所有测试用例文件
def get_testcase_files():
    testcase_pattern = os.path.join(TESTCASES_DIR, '*.py')
    testcase_files = glob.glob(testcase_pattern)
    # 按文件名排序
    testcase_files.sort()
    logger.info(f"Found {len(testcase_files)} test cases: {[os.path.basename(f) for f in testcase_files]}")
    return testcase_files


def load_testcase_module(testcase_file):
    module_name = f"testcase_{os.path.splitext(os.path.basename(testcase_file))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, testcase_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load testcase module from {testcase_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_testcase_spec(testcase_spec, testcase_name):
    if testcase_spec is None:
        raise ValueError(
            f"Testcase {testcase_name} build_testcase() returned None."
        )
    if not isinstance(testcase_spec, dict):
        raise TypeError(
            f"Testcase {testcase_name} build_testcase() returned {type(testcase_spec).__name__}, expected dict."
        )

    required_keys = {"model_or_func", "inputs"}
    missing_keys = sorted(required_keys - testcase_spec.keys())
    if missing_keys:
        raise ValueError(
            f"Testcase {testcase_name} returned an incomplete testcase spec, "
            f"missing keys: {', '.join(missing_keys)}"
        )

    return testcase_spec


def normalize_testcase_spec(testcase_spec, testcase_name):
    normalized_spec = dict(testcase_spec)
    normalized_spec.setdefault("device", "npu")
    normalized_spec.setdefault("warmup_steps", 5)
    normalized_spec.setdefault("exec_steps", 10)
    normalized_spec.setdefault("artifact_subdir", os.path.splitext(testcase_name)[0])
    return normalized_spec


def validate_benchmark_result(raw_result, testcase_name):
    if raw_result is None:
        raise ValueError(
            f"Testcase {testcase_name} benchmark() returned None."
        )
    if not isinstance(raw_result, dict):
        raise TypeError(
            f"Testcase {testcase_name} benchmark() returned {type(raw_result).__name__}, expected dict."
        )

    required_keys = {"precision_passed", "eager_time", "compile_time", "current_time"}
    missing_keys = sorted(required_keys - raw_result.keys())
    if missing_keys:
        raise ValueError(
            f"Testcase {testcase_name} returned an incomplete benchmark result, "
            f"missing keys: {', '.join(missing_keys)}"
        )

    return raw_result


class TestcaseTimeoutError(TimeoutError):
    """Raised when a testcase exceeds the configured execution timeout."""


def _handle_testcase_timeout(signum, frame):
    raise TestcaseTimeoutError


@contextmanager
def testcase_timeout(timeout_seconds):
    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_testcase_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

# 运行单个测试用例
def run_testcase(testcase_file, timeout_seconds=TESTCASE_TIMEOUT_SECONDS, benchmark_runner=None):
    testcase_name = os.path.basename(testcase_file)
    logger.info(f"Running test case: {testcase_name}")

    try:
        with testcase_timeout(timeout_seconds):
            module = load_testcase_module(testcase_file)
            if not hasattr(module, 'build_testcase'):
                raise AttributeError(f"Testcase {testcase_name} does not define build_testcase()")

            testcase_spec = validate_testcase_spec(module.build_testcase(), testcase_name)
            testcase_spec = normalize_testcase_spec(testcase_spec, testcase_name)

            if benchmark_runner is None:
                from utils.benchmark import benchmark as benchmark_runner_impl
            else:
                benchmark_runner_impl = benchmark_runner

            raw_result = validate_benchmark_result(benchmark_runner_impl(**testcase_spec), testcase_name)

        result = {
            'testcase': testcase_name,
            'exit_code': 0,
            'raw_result': raw_result,
            'error_message': '',
        }
        logger.info(f"Test case {testcase_name} completed successfully")
        return result
    except TestcaseTimeoutError:
        error_message = f"Timeout expired after {timeout_seconds} seconds"
        logger.error(f"Test case {testcase_name} timed out after {timeout_seconds} seconds")
        return {
            'testcase': testcase_name,
            'exit_code': -1,
            'raw_result': None,
            'error_message': error_message,
        }
    except Exception as e:
        logger.exception(f"Error running test case {testcase_name}")
        return {
            'testcase': testcase_name,
            'exit_code': 1,
            'raw_result': None,
            'error_message': str(e),
        }


def main(argv=None):
    from utils import setup_logging
    from utils.profiler import cleanup_directories

    args = parse_args(argv)
    setup_logging()
    logger.info("Starting case evaluation...")

    try:
        # 获取所有测试用例
        testcase_files = get_testcase_files()

        if not testcase_files:
            logger.warning("No test cases found in testcases directory")
            final_result = build_empty_result()
            logger.info(f"final_result json: {json.dumps(final_result, indent=2)}")
            logger.info("Case evaluation completed")
            return final_result

        # 运行所有测试用例
        testcase_results = []
        for testcase_file in testcase_files:
            result = run_testcase(testcase_file)
            testcase_results.append(result)

        # 生成最终结果
        baseline_data = load_baseline_data(BASELINE_DATA_FILE)
        final_result = generate_final_result(
            testcase_results,
            baseline_data
        )

        logger.info(f"final_result json: {json.dumps(final_result, indent=2)}")
        logger.info("Case evaluation completed")
        return final_result
    finally:
        if args.clean_up:
            try:
                cleanup_directories()
            except Exception:
                logger.exception("Failed to clean up generated artifact directories")

if __name__ == '__main__':
    main()
