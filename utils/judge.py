import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

FUNCTIONAL_PASS_TOKEN = "Precision test result: Passed"
EAGER_TIME_PATTERN = re.compile(r"\[eager\] Average execution time:\s*([\d.]+)\s*us")
COMPILE_TIME_PATTERN = re.compile(r"\[compile\] Average execution time:\s*([\d.]+)\s*us")
CURRENT_TIME_PATTERN = re.compile(r"\[current\] Average execution time:\s*([\d.]+)\s*us")


def load_baseline_data(baseline_data_file):
    """Load baseline data from yaml."""
    return {}


def is_functional_test_passed(stdout):
    """Return whether the testcase output indicates a functional pass."""
    return FUNCTIONAL_PASS_TOKEN in stdout if stdout else False


def get_testcase_text(testcase_result):
    """Return combined testcase output text from stdout and stderr."""
    parts = [testcase_result.get("stdout", ""), testcase_result.get("stderr", "")]
    return "\n".join(part for part in parts if part)


def parse_testcase_output(testcase_result):
    """Parse testcase output and extract performance metrics (eager, compile, and current)."""
    output_text = get_testcase_text(testcase_result)
    parsed_data = {
        "functional_passed": is_functional_test_passed(output_text),
        "eager_time": None,
        "compile_time": None,
        "current_time": None,
    }

    match_eager = EAGER_TIME_PATTERN.search(output_text)
    if match_eager:
        try:
            parsed_data["eager_time"] = float(match_eager.group(1))
        except ValueError:
            logger.warning("Failed to parse eager_time from: %s", match_eager.group(1))

    match_compile = COMPILE_TIME_PATTERN.search(output_text)
    if match_compile:
        try:
            parsed_data["compile_time"] = float(match_compile.group(1))
        except ValueError:
            logger.warning("Failed to parse compile_time from: %s", match_compile.group(1))

    match_current = CURRENT_TIME_PATTERN.search(output_text)
    if match_current:
        try:
            parsed_data["current_time"] = float(match_current.group(1))
        except ValueError:
            logger.warning("Failed to parse current_time from: %s", match_current.group(1))

    logger.info("Parsed testcase - functional_passed: %s, eager_time: %s us, compile_time: %s us, current_time: %s us",
                parsed_data["functional_passed"], parsed_data["eager_time"], parsed_data["compile_time"], parsed_data["current_time"])
    return parsed_data


def calculate_testcase_score(testcase_result, parsed_output, baseline_data):
    """
    Calculate testcase performance metrics based on the parsed output.
    Returns s_i (performance term) and b1_i (eager time for weight calculation).
    According to the score.md document:
    - s_i = (b2_i / current_i) * I_i, where I_i is 1 if functional_passed else 0
    - b1_i is eager_time (for weight calculation)
    - b2_i is compile_time (baseline for performance ratio)
    - current_i is current_time (user's implementation)
    """
    testcase_name = testcase_result["testcase"]

    if testcase_result["exit_code"] != 0:
        logger.error(
            "Test case %s failed with exit code %s, s_i: 0.0",
            testcase_name,
            testcase_result["exit_code"],
        )
        return 0.0, 0.0

    functional_passed = parsed_output.get("functional_passed", False)
    eager_time = parsed_output.get("eager_time")
    compile_time = parsed_output.get("compile_time")
    current_time = parsed_output.get("current_time")

    I_i = 1.0 if functional_passed else 0.0

    if not functional_passed:
        logger.error("Test case %s functional test failed, s_i: 0.0", testcase_name)
        return 0.0, eager_time if eager_time is not None else 0.0

    if compile_time is None or compile_time <= 0:
        logger.error("Invalid compile_time (b2_i) for %s, s_i: 0.0", testcase_name)
        return 0.0, eager_time if eager_time is not None else 0.0

    if current_time is None or current_time <= 0:
        logger.error("Invalid current_time for %s, s_i: 0.0", testcase_name)
        return 0.0, eager_time if eager_time is not None else 0.0

    s_i = (compile_time / current_time) * I_i
    logger.info(
        "Performance term for %s: %.4f (b2_i: %s us, current_i: %s us, I_i: %s)",
        testcase_name,
        s_i,
        compile_time,
        current_time,
        I_i,
    )

    return s_i, eager_time if eager_time is not None else 0.0


def generate_final_result(testcase_results, baseline_data, now=None):
    """Generate the final result from all testcase results according to score.md."""
    timestamp = (now or datetime.now()).isoformat()
    all_passed = all(result["exit_code"] == 0 for result in testcase_results)
    verdict = "AC" if all_passed else "WA"

    parsed_outputs = []
    testcase_s_i = []  # s_i for each test case
    testcase_b1_i = []  # b1_i (eager_time) for weight calculation
    functional_indicators = []  # I_i for each test case

    total_b1 = 0.0  # sum of b1_i (eager_time) for all test cases

    for result in testcase_results:
        parsed_output = parse_testcase_output(result)
        s_i, b1_i = calculate_testcase_score(result, parsed_output, baseline_data)

        parsed_outputs.append(parsed_output)
        testcase_s_i.append(s_i)
        testcase_b1_i.append(b1_i)
        I_i = 1.0 if parsed_output.get("functional_passed", False) else 0.0
        functional_indicators.append(I_i)

        if b1_i is not None:
            total_b1 += b1_i

    n = len(testcase_results)
    F = sum(functional_indicators) / n if n > 0 else 0.0

    sum_I_w = 0.0  # sum I_i * w_i
    sum_I_w_s = 0.0  # sum I_i * w_i * s_i

    for i in range(n):
        I_i = functional_indicators[i]
        b1_i = testcase_b1_i[i]
        w_i = b1_i / total_b1 if total_b1 > 0 else 0.0  # w_i = b1_i / sum(b1_j)
        s_i = testcase_s_i[i]

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
        "testcase_details": [
            {
                "testcase": result["testcase"],
                "exit_code": result["exit_code"],
                "functional_passed": parsed_outputs[index].get("functional_passed", False),
                "s_i": testcase_s_i[index],
                "eager_time": parsed_outputs[index].get("eager_time"),
                "compile_time": parsed_outputs[index].get("compile_time"),
                "current_time": parsed_outputs[index].get("current_time"),
                "weight": testcase_b1_i[index] / total_b1 if total_b1 > 0 else 0.0,
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
