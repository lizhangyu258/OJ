import logging
import os
import re
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)

FUNCTIONAL_PASS_TOKEN = "Precision test result: Passed"
AVG_EXEC_TIME_PATTERN = re.compile(r"Average execution time:\s*([\d.]+)\s*us")


def load_baseline_data(baseline_data_file):
    """Load baseline data from yaml."""
    if not os.path.exists(baseline_data_file):
        logger.warning("Baseline data file not found: %s", baseline_data_file)
        return {}

    try:
        with open(baseline_data_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except Exception as exc:
        logger.error("Error loading baseline data: %s", exc)
        return {}

    logger.info("Loaded baseline data from %s", baseline_data_file)
    return data if data else {}


def is_functional_test_passed(stdout):
    """Return whether the testcase output indicates a functional pass."""
    return FUNCTIONAL_PASS_TOKEN in stdout if stdout else False


def parse_testcase_output(testcase_result):
    """Parse testcase output and extract performance metrics."""
    stdout = testcase_result.get("stdout", "")
    parsed_data = {
        "avg_execution_time": None,
        "functional_passed": is_functional_test_passed(stdout),
    }

    match = AVG_EXEC_TIME_PATTERN.search(stdout)
    if not match:
        logger.warning("Average execution time not found in output")
        return parsed_data

    try:
        avg_time = float(match.group(1))
    except ValueError:
        logger.warning("Failed to parse average execution time from: %s", match.group(1))
        return parsed_data

    parsed_data["avg_execution_time"] = avg_time
    logger.info("Parsed average execution time: %s us", avg_time)
    return parsed_data


def calculate_testcase_score(testcase_result, parsed_output, baseline_data):
    """Calculate testcase score based on the parsed output and baseline data."""
    testcase_name = testcase_result["testcase"]

    if testcase_result["exit_code"] != 0:
        logger.error(
            "Test case %s failed with exit code %s, score: 0.0",
            testcase_name,
            testcase_result["exit_code"],
        )
        return 0.0

    current_avg_time = parsed_output.get("avg_execution_time")
    if current_avg_time is None:
        logger.error("No average execution time found for %s, score: 0.0", testcase_name)
        return 0.0

    baseline_entry = baseline_data.get(testcase_name)
    if not isinstance(baseline_entry, dict):
        logger.error("No baseline data found for %s, score: 0.0", testcase_name)
        return 0.0

    baseline_avg_time = baseline_entry.get("avg_execution_time")
    if baseline_avg_time is None or baseline_avg_time <= 0:
        logger.error("Invalid baseline avg_execution_time for %s, score: 0.0", testcase_name)
        return 0.0

    speedup_ratio = baseline_avg_time / current_avg_time
    logger.info(
        "Speedup ratio for %s: %.4f (baseline: %s us, current: %s us)",
        testcase_name,
        speedup_ratio,
        baseline_avg_time,
        current_avg_time,
    )
    return speedup_ratio


def generate_final_result(testcase_results, baseline_data, now=None):
    """Generate the final result from all testcase results."""
    timestamp = (now or datetime.now()).isoformat()
    all_passed = all(result["exit_code"] == 0 for result in testcase_results)
    verdict = "AC" if all_passed else "WA"

    total_score = 0.0
    parsed_outputs = []
    testcase_scores = []

    for result in testcase_results:
        parsed_output = parse_testcase_output(result)
        score = calculate_testcase_score(result, parsed_output, baseline_data)
        logger.info("Test case %s score: %.4f", result["testcase"], score)

        parsed_outputs.append(parsed_output)
        testcase_scores.append(score)
        total_score += score

    avg_score = total_score / len(testcase_results) if testcase_results else 0.0

    detail = {
        "timestamp": timestamp,
        "total_testcases": len(testcase_results),
        "passed_testcases": sum(1 for result in testcase_results if result["exit_code"] == 0),
        "failed_testcases": sum(1 for result in testcase_results if result["exit_code"] != 0),
        "testcase_details": [
            {
                "testcase": result["testcase"],
                "exit_code": result["exit_code"],
                "score": testcase_scores[index],
                "avg_execution_time": parsed_outputs[index].get("avg_execution_time"),
            }
            for index, result in enumerate(testcase_results)
        ],
    }

    return {
        "verdict": verdict,
        "rank": {
            "rank": avg_score,
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
            "testcase_details": [],
        },
    }
