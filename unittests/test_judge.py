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
from utils.profiler import get_step_time_from_csv


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
[current] Average execution time: 40.0 us
""",
        }

        parsed = parse_testcase_output(testcase_result)

        self.assertTrue(parsed["functional_passed"])
        self.assertEqual(parsed["eager_time"], 100.0)
        self.assertEqual(parsed["compile_time"], 50.0)
        self.assertEqual(parsed["current_time"], 40.0)

    def test_parse_testcase_output_handles_missing_times(self):
        testcase_result = {
            "stdout": "header\nPrecision test result: Failed\n",
        }

        parsed = parse_testcase_output(testcase_result)

        self.assertIsNone(parsed["eager_time"])
        self.assertIsNone(parsed["compile_time"])
        self.assertIsNone(parsed["current_time"])
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
        parsed_output = {"eager_time": 100.0, "compile_time": 50.0, "current_time": 40.0, "functional_passed": True}
        baseline_data = {}

        s_i, b1_i = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(s_i, 0.0)
        self.assertEqual(b1_i, 0.0)

    def test_calculate_testcase_score_returns_zero_when_functional_failed(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"eager_time": 100.0, "compile_time": 50.0, "current_time": 40.0, "functional_passed": False}
        baseline_data = {}

        s_i, b1_i = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(s_i, 0.0)
        self.assertEqual(b1_i, 100.0)

    def test_calculate_testcase_score_returns_s_i_according_to_formula(self):
        testcase_result = {"testcase": "case1.py", "exit_code": 0}
        parsed_output = {"eager_time": 100.0, "compile_time": 50.0, "current_time": 40.0, "functional_passed": True}
        baseline_data = {}

        s_i, b1_i = calculate_testcase_score(testcase_result, parsed_output, baseline_data)

        self.assertEqual(s_i, 50.0 / 40.0)
        self.assertEqual(b1_i, 100.0)

    def test_generate_final_result_with_full_scoring_mechanism(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 100.0 us
[compile] Average execution time: 50.0 us
[current] Average execution time: 40.0 us
""",
                "stderr": "",
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 200.0 us
[compile] Average execution time: 100.0 us
[current] Average execution time: 50.0 us
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
            testcase_results, baseline_data, now=now
        )

        self.assertEqual(final_result["verdict"], "WA")
        expected_final = 0.4 * (2/3) + 0.6 * (( (100/300)*(50/40) + (200/300)*(100/50) ) / (100/300 + 200/300))
        self.assertAlmostEqual(final_result["rank"]["rank"], expected_final)
        self.assertEqual(final_result["detail"]["timestamp"], now.isoformat())
        self.assertEqual(final_result["detail"]["passed_testcases"], 2)
        self.assertEqual(final_result["detail"]["failed_testcases"], 1)
        self.assertEqual(final_result["detail"]["functional_score"], 2/3)
        expected_performance = (( (100/300)*(50/40) + (200/300)*(100/50) ) / (100/300 + 200/300))
        self.assertAlmostEqual(final_result["detail"]["performance_score"], expected_performance)
        self.assertEqual(final_result["detail"]["testcase_details"][0]["s_i"], 50.0/40.0)
        self.assertEqual(final_result["detail"]["testcase_details"][1]["s_i"], 100.0/50.0)
        self.assertEqual(final_result["detail"]["testcase_details"][2]["s_i"], 0.0)

    def test_generate_final_result_with_weighted_average(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 100.0 us
[compile] Average execution time: 50.0 us
[current] Average execution time: 40.0 us
""",
                "stderr": "",
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "stdout": """Precision test result: Passed
[eager] Average execution time: 300.0 us
[compile] Average execution time: 100.0 us
[current] Average execution time: 50.0 us
""",
                "stderr": "",
            },
        ]
        baseline_data = {}
        now = datetime(2026, 4, 16, 12, 0, 0)

        final_result = generate_final_result(
            testcase_results, baseline_data, now=now
        )

        expected_performance = ( (0.25 * 1.25 + 0.75 * 2.0) / (0.25 + 0.75) )
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


class ProfilerTests(unittest.TestCase):
    def test_get_step_time_from_csv_returns_none_when_file_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.csv"
            self.assertIsNone(get_step_time_from_csv(str(missing_path)))

    def test_get_step_time_from_csv_returns_none_when_only_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("header1,header2,header3,header4\n", encoding="utf-8")
            
            self.assertIsNone(get_step_time_from_csv(str(csv_path)))

    def test_get_step_time_from_csv_calculates_average_single_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("""header1,header2,header3,header4,header5
col1,col2,100.0,col4,col5
""", encoding="utf-8")
            
            result = get_step_time_from_csv(str(csv_path))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, 100.0)

    def test_get_step_time_from_csv_calculates_average_multiple_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("""header1,header2,header3,header4,header5
col1,col2,100.0,col4,col5
col1,col2,200.0,col4,col5
col1,col2,300.0,col4,col5
""", encoding="utf-8")
            
            result = get_step_time_from_csv(str(csv_path))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, 200.0)

    def test_get_step_time_from_csv_returns_none_when_no_valid_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("""header1,header2,header3,header4,header5
col1,col2
col1,col2,abc,col4,col5
""", encoding="utf-8")
            
            self.assertIsNone(get_step_time_from_csv(str(csv_path)))


if __name__ == "__main__":
    unittest.main()
