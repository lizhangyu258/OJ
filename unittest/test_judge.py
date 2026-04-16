import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.judge import build_empty_result
from utils.judge import calculate_testcase_score
from utils.judge import generate_final_result
from utils.judge import is_functional_test_passed
from utils.judge import load_baseline_data
from utils.judge import parse_testcase_output


class CaseJudgeCoreTests(unittest.TestCase):
    def test_is_functional_test_passed_returns_true_when_pass_token_exists(self):
        stdout = "compile ok\nPrecision test result: Passed\n"
        self.assertTrue(is_functional_test_passed(stdout))

    def test_is_functional_test_passed_returns_false_when_pass_token_missing(self):
        stdout = "compile ok\nPrecision test result: Failed\n"
        self.assertFalse(is_functional_test_passed(stdout))

    def test_parse_testcase_output_extracts_eager_and_compile_times(self):
        testcase_result = {
            "stdout": """Precision test result: Passed
[eager] Average execution time: 100.0 us
[compile] Average execution time: 50.0 us
""",
        }

        parsed = parse_testcase_output(testcase_result)

        self.assertTrue(parsed["functional_passed"])
        self.assertEqual(parsed["eager_time"], 100.0)
        self.assertEqual(parsed["compile_time"], 50.0)

    def test_parse_testcase_output_handles_missing_times(self):
        testcase_result = {
            "stdout": "header\nPrecision test result: Failed\n",
        }

        parsed = parse_testcase_output(testcase_result)

        self.assertIsNone(parsed["eager_time"])
        self.assertIsNone(parsed["compile_time"])
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

            self.assertEqual(baseline, {})

    def test_calculate_testcase_score_returns_zero_when_exit_code_non_zero(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 1}
        parsed_output = {"eager_time": 100.0, "compile_time": 50.0, "functional_passed": True}
        baseline_data = {}

        speedup, weight = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(speedup, 0.0)
        self.assertEqual(weight, 0.0)

    def test_calculate_testcase_score_returns_zero_when_functional_failed(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"eager_time": 100.0, "compile_time": 50.0, "functional_passed": False}
        baseline_data = {}

        speedup, weight = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(speedup, 0.0)
        self.assertEqual(weight, 0.0)

    def test_calculate_testcase_score_returns_speedup_using_eager(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"eager_time": 100.0, "compile_time": 40.0, "functional_passed": True}
        baseline_data = {}

        speedup, weight = calculate_testcase_score(
            testcase_result, parsed_output, baseline_data, use_baseline_eager=True
        )

        self.assertEqual(speedup, 2.5)
        self.assertEqual(weight, 100.0)

    def test_calculate_testcase_score_returns_speedup_using_compile(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"eager_time": 100.0, "compile_time": 40.0, "functional_passed": True}
        baseline_data = {}

        speedup, weight = calculate_testcase_score(
            testcase_result, parsed_output, baseline_data, use_baseline_eager=False
        )

        self.assertEqual(speedup, 1.0)
        self.assertEqual(weight, 100.0)

    def test_generate_final_result_with_full_scoring_mechanism(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 100.0 us
[compile] Average execution time: 50.0 us
""",
                "stderr": "",
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 200.0 us
[compile] Average execution time: 100.0 us
""",
                "stderr": "",
            },
            {
                "testcase": "case3.py",
                "exit_code": 1,
                "stdout": "Precision test result: Failed\n",
                "stderr": "runtime error",
            },
        ]
        baseline_data = {}
        now = datetime(2026, 4, 16, 12, 0, 0)

        final_result = generate_final_result(
            testcase_results, baseline_data, now=now, use_baseline_eager=True
        )

        self.assertEqual(final_result["verdict"], "WA")
        expected_final = 0.4 * (2/3) + 0.6 * 2.0
        self.assertAlmostEqual(final_result["rank"]["rank"], expected_final)
        self.assertEqual(final_result["detail"]["timestamp"], now.isoformat())
        self.assertEqual(final_result["detail"]["passed_testcases"], 2)
        self.assertEqual(final_result["detail"]["failed_testcases"], 1)
        self.assertEqual(final_result["detail"]["functional_score"], 2/3)
        self.assertAlmostEqual(final_result["detail"]["performance_score"], 2.0)
        self.assertEqual(final_result["detail"]["testcase_details"][0]["speedup"], 2.0)
        self.assertEqual(final_result["detail"]["testcase_details"][1]["speedup"], 2.0)
        self.assertEqual(final_result["detail"]["testcase_details"][2]["speedup"], 0.0)

    def test_generate_final_result_with_weighted_average(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 100.0 us
[compile] Average execution time: 50.0 us
""",
                "stderr": "",
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 300.0 us
[compile] Average execution time: 100.0 us
""",
                "stderr": "",
            },
        ]
        baseline_data = {}
        now = datetime(2026, 4, 16, 12, 0, 0)

        final_result = generate_final_result(
            testcase_results, baseline_data, now=now, use_baseline_eager=True
        )

        expected_performance = (0.25 * 2.0 + 0.75 * 3.0) / (0.25 + 0.75)
        self.assertAlmostEqual(final_result["detail"]["performance_score"], expected_performance)
        expected_final = 0.4 * 1.0 + 0.6 * expected_performance
        self.assertAlmostEqual(final_result["rank"]["rank"], expected_final)

    def test_generate_final_result_all_functional_failed(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 1,
                "stdout": "Error\n",
                "stderr": "runtime error",
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "stdout": "Precision test result: Failed\n",
                "stderr": "",
            },
        ]
        baseline_data = {}
        now = datetime(2026, 4, 16, 12, 0, 0)

        final_result = generate_final_result(testcase_results, baseline_data, now=now)

        self.assertEqual(final_result["detail"]["functional_score"], 0.0)
        self.assertEqual(final_result["detail"]["performance_score"], 0.0)
        self.assertEqual(final_result["rank"]["rank"], 0.0)

    def test_build_empty_result_uses_expected_shape(self):
        now = datetime(2026, 4, 16, 12, 0, 0)

        result = build_empty_result(now=now)

        self.assertEqual(result["verdict"], "WA")
        self.assertEqual(result["rank"]["rank"], 0.0)
        self.assertEqual(result["detail"]["timestamp"], now.isoformat())
        self.assertEqual(result["detail"]["testcase_details"], [])


if __name__ == "__main__":
    unittest.main()
