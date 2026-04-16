import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from utils.judge import build_empty_result
from utils.judge import calculate_testcase_score
from utils.judge import generate_final_result
from utils.judge import is_functional_test_passed
from utils.judge import load_baseline_data
from utils.judge import parse_testcase_output


class CaseJudgeCoreTests(unittest.TestCase):
    def test_is_functional_test_passed_returns_true_when_pass_token_exists(self):
        stdout = "compile ok\nPrecision test result: Passed\nAverage execution time: 12.5 us\n"
        self.assertTrue(is_functional_test_passed(stdout))

    def test_is_functional_test_passed_returns_false_when_pass_token_missing(self):
        stdout = "compile ok\nPrecision test result: Failed\n"
        self.assertFalse(is_functional_test_passed(stdout))

    def test_parse_testcase_output_extracts_avg_time_and_functional_status(self):
        testcase_result = {
            "stdout": "header\nPrecision test result: Passed\nAverage execution time: 12.5 us\n",
        }

        parsed = parse_testcase_output(testcase_result)

        self.assertEqual(parsed["avg_execution_time"], 12.5)
        self.assertTrue(parsed["functional_passed"])

    def test_parse_testcase_output_handles_missing_avg_time(self):
        testcase_result = {
            "stdout": "header\nPrecision test result: Failed\n",
        }

        parsed = parse_testcase_output(testcase_result)

        self.assertIsNone(parsed["avg_execution_time"])
        self.assertFalse(parsed["functional_passed"])

    def test_load_baseline_data_returns_empty_dict_when_file_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = Path(temp_dir) / "missing.yaml"
            self.assertEqual(load_baseline_data(str(baseline_path)), {})

    def test_load_baseline_data_reads_yaml_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = Path(temp_dir) / "data.yaml"
            baseline_path.write_text("case1.py:\n  avg_execution_time: 20.0\n", encoding="utf-8")

            baseline = load_baseline_data(str(baseline_path))

            self.assertEqual(baseline["case1.py"]["avg_execution_time"], 20.0)

    def test_calculate_testcase_score_returns_zero_when_exit_code_non_zero(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 1}
        parsed_output = {"avg_execution_time": 10.0}
        baseline_data = {"case1.py": {"avg_execution_time": 20.0}}

        score = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(score, 0.0)

    def test_calculate_testcase_score_returns_zero_when_baseline_missing(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"avg_execution_time": 10.0}

        score = calculate_testcase_score(testcase_result, parsed_output, {})

        self.assertEqual(score, 0.0)

    def test_calculate_testcase_score_returns_speedup_ratio(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"avg_execution_time": 10.0}
        baseline_data = {"case1.py": {"avg_execution_time": 20.0}}

        score = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(score, 2.0)

    def test_generate_final_result_aggregates_case_scores(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 0,
                "stdout": "Precision test result: Passed\nAverage execution time: 10.0 us\n",
                "stderr": "",
            },
            {
                "testcase": "case2.py",
                "exit_code": 1,
                "stdout": "Precision test result: Failed\n",
                "stderr": "runtime error",
            },
        ]
        baseline_data = {
            "case1.py": {"avg_execution_time": 20.0},
            "case2.py": {"avg_execution_time": 30.0},
        }
        now = datetime(2026, 4, 16, 12, 0, 0)

        final_result = generate_final_result(testcase_results, baseline_data, now=now)

        self.assertEqual(final_result["verdict"], "WA")
        self.assertEqual(final_result["rank"]["rank"], 1.0)
        self.assertEqual(final_result["detail"]["timestamp"], now.isoformat())
        self.assertEqual(final_result["detail"]["passed_testcases"], 1)
        self.assertEqual(final_result["detail"]["failed_testcases"], 1)
        self.assertEqual(final_result["detail"]["testcase_details"][0]["score"], 2.0)
        self.assertEqual(final_result["detail"]["testcase_details"][1]["score"], 0.0)

    def test_build_empty_result_uses_expected_shape(self):
        now = datetime(2026, 4, 16, 12, 0, 0)

        result = build_empty_result(now=now)

        self.assertEqual(result["verdict"], "WA")
        self.assertEqual(result["rank"]["rank"], 0.0)
        self.assertEqual(result["detail"]["timestamp"], now.isoformat())
        self.assertEqual(result["detail"]["testcase_details"], [])


if __name__ == "__main__":
    unittest.main()
