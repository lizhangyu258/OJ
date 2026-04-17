import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_baseline_data(baseline_data_file):
    """Load baseline data from yaml."""
    return {}


def _coerce_optional_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Failed to convert %r to float", value)
        return None


def build_serializable_benchmark_result(benchmark_result):
    """Keep only the benchmark fields needed by the judge."""
    if benchmark_result is None:
        return None

    return {
        "precision_passed": bool(benchmark_result.get("precision_passed", False)),
        "eager_time": _coerce_optional_float(benchmark_result.get("eager_time")),
        "compile_time": _coerce_optional_float(benchmark_result.get("compile_time")),
        "current_time": _coerce_optional_float(benchmark_result.get("current_time")),
        "max_diff": _coerce_optional_float(benchmark_result.get("max_diff")),
        "speedup": _coerce_optional_float(benchmark_result.get("speedup")),
    }


def parse_testcase_output(testcase_result):
    """Extract testcase metrics from structured benchmark data."""
    benchmark_result = build_serializable_benchmark_result(testcase_result.get("benchmark_result"))
    parsed_data = {
        "functional_passed": benchmark_result["precision_passed"] if benchmark_result is not None else False,
        "eager_time": benchmark_result["eager_time"] if benchmark_result is not None else None,
        "compile_time": benchmark_result["compile_time"] if benchmark_result is not None else None,
        "current_time": benchmark_result["current_time"] if benchmark_result is not None else None,
    }
    logger.info(
        "Parsed testcase result - functional_passed: %s, eager_time: %s us, compile_time: %s us, current_time: %s us",
        parsed_data["functional_passed"],
        parsed_data["eager_time"],
        parsed_data["compile_time"],
        parsed_data["current_time"],
    )
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
    passed_testcases = sum(
        1
        for result, parsed_output in zip(testcase_results, parsed_outputs)
        if result["exit_code"] == 0 and parsed_output.get("functional_passed", False)
    )
    failed_testcases = n - passed_testcases

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
    verdict = "AC" if failed_testcases == 0 else "WA"

    detail = {
        "timestamp": timestamp,
        "total_testcases": n,
        "passed_testcases": passed_testcases,
        "failed_testcases": failed_testcases,
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
