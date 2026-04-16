import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

FUNCTIONAL_PASS_TOKEN = "Precision test result: Passed"
EAGER_TIME_PATTERN = re.compile(r"\[eager\] Average execution time:\s*([\d.]+)\s*us")
COMPILE_TIME_PATTERN = re.compile(r"\[compile\] Average execution time:\s*([\d.]+)\s*us")


def load_baseline_data(baseline_data_file):
    """Load baseline data from yaml."""
    return {}


def is_functional_test_passed(stdout):
    """Return whether the testcase output indicates a functional pass."""
    return FUNCTIONAL_PASS_TOKEN in stdout if stdout else False


def parse_testcase_output(testcase_result):
    """Parse testcase output and extract performance metrics (eager and compile)."""
    stdout = testcase_result.get("stdout", "")
    parsed_data = {
        "functional_passed": is_functional_test_passed(stdout),
        "eager_time": None,
        "compile_time": None,
    }

    match_eager = EAGER_TIME_PATTERN.search(stdout)
    if match_eager:
        try:
            parsed_data["eager_time"] = float(match_eager.group(1))
        except ValueError:
            logger.warning("Failed to parse eager_time from: %s", match_eager.group(1))

    match_compile = COMPILE_TIME_PATTERN.search(stdout)
    if match_compile:
        try:
            parsed_data["compile_time"] = float(match_compile.group(1))
        except ValueError:
            logger.warning("Failed to parse compile_time from: %s", match_compile.group(1))

    logger.info("Parsed testcase - functional_passed: %s, eager_time: %s us, compile_time: %s us",
                parsed_data["functional_passed"], parsed_data["eager_time"], parsed_data["compile_time"])
    return parsed_data


def calculate_testcase_score(testcase_result, parsed_output, baseline_data, use_baseline_eager=True):
    """
    Calculate testcase speedup based on the parsed output using dynamically collected eager and compile.
    Returns the speedup ratio r_i = baseline / current, where current is compile_time.
    """
    testcase_name = testcase_result["testcase"]

    if testcase_result["exit_code"] != 0:
        logger.error(
            "Test case %s failed with exit code %s, speedup: 0.0",
            testcase_name,
            testcase_result["exit_code"],
        )
        return 0.0, 0.0

    functional_passed = parsed_output.get("functional_passed", False)
    eager_time = parsed_output.get("eager_time")
    compile_time = parsed_output.get("compile_time")

    if not functional_passed:
        logger.error("Test case %s functional test failed, speedup: 0.0", testcase_name)
        return 0.0, 0.0

    if compile_time is None or compile_time <= 0:
        logger.error("Invalid compile_time for %s, speedup: 0.0", testcase_name)
        return 0.0, 0.0

    baseline_time = eager_time if use_baseline_eager else compile_time
    if baseline_time is None or baseline_time <= 0:
        logger.error("Invalid baseline time for %s, speedup: 0.0", testcase_name)
        return 0.0, 0.0

    speedup_ratio = baseline_time / compile_time
    logger.info(
        "Speedup ratio for %s: %.4f (baseline: %s us, current: %s us, use_eager: %s)",
        testcase_name,
        speedup_ratio,
        baseline_time,
        compile_time,
        use_baseline_eager,
    )

    return speedup_ratio, eager_time if eager_time is not None else 0.0


def generate_final_result(testcase_results, baseline_data, now=None, use_baseline_eager=True):
    """Generate the final result from all testcase results with dynamic baseline collection."""
    timestamp = (now or datetime.now()).isoformat()
    all_passed = all(result["exit_code"] == 0 for result in testcase_results)
    verdict = "AC" if all_passed else "WA"

    parsed_outputs = []
    testcase_speedups = []
    testcase_weights = []
    functional_indicators = []

    total_eager = 0.0

    for result in testcase_results:
        parsed_output = parse_testcase_output(result)
        speedup, eager_time = calculate_testcase_score(
            result, parsed_output, baseline_data, use_baseline_eager=use_baseline_eager
        )

        parsed_outputs.append(parsed_output)
        testcase_speedups.append(speedup)
        testcase_weights.append(eager_time)
        I_i = 1.0 if parsed_output.get("functional_passed", False) else 0.0
        functional_indicators.append(I_i)

        if eager_time is not None:
            total_eager += eager_time

    n = len(testcase_results)
    F = sum(functional_indicators) / n if n > 0 else 0.0

    sum_I_w = 0.0
    sum_I_w_s = 0.0

    for i in range(n):
        I_i = functional_indicators[i]
        w_i = testcase_weights[i] / total_eager if total_eager > 0 else 0.0
        s_i = testcase_speedups[i]

        sum_I_w += I_i * w_i
        sum_I_w_s += I_i * w_i * s_i

    P = sum_I_w_s / sum_I_w if sum_I_w > 0 else 0.0

    S = 0.4 * F + 0.6 * P

    detail = {
        "timestamp": timestamp,
        "total_testcases": n,
        "passed_testcases": sum(1 for result in testcase_results if result["exit_code"] == 0),
        "failed_testcases": sum(1 for result in testcase_results if result["exit_code"] != 0),
        "functional_score": F,
        "performance_score": P,
        "final_score": S,
        "use_baseline_eager": use_baseline_eager,
        "testcase_details": [
            {
                "testcase": result["testcase"],
                "exit_code": result["exit_code"],
                "functional_passed": parsed_outputs[index].get("functional_passed", False),
                "speedup": testcase_speedups[index],
                "eager_time": parsed_outputs[index].get("eager_time"),
                "compile_time": parsed_outputs[index].get("compile_time"),
                "weight": testcase_weights[index] / total_eager if total_eager > 0 else 0.0,
            }
            for index, result in enumerate(testcase_results)
        ],
    }

    return {
        "verdict": verdict,
        "rank": {
            "rank": S,
        },
        "detail": detail,
    }


def build_empty_result(now=None):
    """Build the result for the empty-testcase case."""
    return {
        "verdict": "WA",
        "rank": {"rank": 0.0},
        "detail": {
            "timestamp": (now or datetime.now()).isoformat(),
            "total_testcases": 0,
            "passed_testcases": 0,
            "failed_testcases": 0,
            "functional_score": 0.0,
            "performance_score": 0.0,
            "final_score": 0.0,
            "testcase_details": [],
        },
    }
