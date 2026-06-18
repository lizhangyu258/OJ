import io
import json
import os
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from case_judge import parse_args
from case_judge import emit_result
from case_judge import format_platform_result
from case_judge import load_bin_config
from case_judge import run_testcase
from case_judge import validate_bin_config_tools
from case_judge import validate_final_result_metrics
from utils.judge import build_empty_result
from utils.judge import calculate_testcase_score
from utils.judge import extract_testcase_metrics
from utils.judge import generate_final_result
from utils.judge import load_baseline_data
from utils.binary_manager import copy_tools_to_binary
from utils.binary_manager import tool_binary_context
from utils.binary_manager import _clear_binary_dir
from utils.binary_manager import _ensure_binary_dir
from utils.binary_manager import _get_binary_dir
from utils.binary_manager import _verify_binary_resolves
from utils.tool_validation import ToolValidationError
from utils.tool_validation import IllegalToolBinaryError
from utils.profiler import cleanup_directories
from utils.profiler import get_step_time_from_csv
from utils.profiler import resolve_output_dir
from utils.profiler import setup_profiler_output


def write_fake_tool(path: Path):
    path.write_text("#!/bin/sh\necho 'hfusion-auto-schedule'\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


class CaseJudgeCoreTests(unittest.TestCase):
    def test_format_platform_result_serializes_detail_to_string(self):
        result = {
            "verdict": "WA",
            "rank": {"rank": 0.0},
            "detail": {"failed_testcases": 1},
        }

        platform_result = format_platform_result(result)

        self.assertIsInstance(platform_result["detail"], str)
        self.assertEqual(json.loads(platform_result["detail"]), {"failed_testcases": 1})

    def test_emit_result_prints_detail_as_string(self):
        result = {
            "verdict": "WA",
            "rank": {"rank": 0.0},
            "detail": {"failed_testcases": 1},
        }
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            emit_result(result)

        emitted = json.loads(stdout.getvalue())
        self.assertIsInstance(emitted["detail"], str)

    def test_format_platform_result_keeps_string_detail(self):
        result = {
            "verdict": "CE",
            "rank": {"rank": -1},
            "detail": "traceback",
        }

        platform_result = format_platform_result(result)

        self.assertEqual(platform_result["detail"], "traceback")

    def test_parse_args_defaults_clean_up_to_false(self):
        args = parse_args([])

        self.assertFalse(args.clean_up)

    def test_parse_args_enables_clean_up_option(self):
        args = parse_args(["--clean-up"])

        self.assertTrue(args.clean_up)

    def test_load_bin_config_reads_bin_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                """bin:
  baseline: /opt/baseline/bin
  current: /opt/current/bin
""",
                encoding="utf-8",
            )

            self.assertEqual(
                load_bin_config(str(config_path)),
                {
                    "baseline": "/opt/baseline/bin",
                    "current": "/opt/current/bin",
                },
            )

    def test_load_bin_config_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            self.assertIsNone(load_bin_config(str(config_path)))

    def test_check_bin_path_script_validates_current_bin_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bin_dir = temp_path / "bin"
            bin_dir.mkdir()
            write_fake_tool(bin_dir / "bishengir-compile")
            write_fake_tool(bin_dir / "bishengir-opt")
            (temp_path / "config.yaml").write_text(
                f"""bin:
  current: {bin_dir}
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "/Users/ayame/workspace/OJ/check_bin_path.py",
                    "--config",
                    str(temp_path / "config.yaml"),
                    "--key",
                    "current",
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(os.path.realpath(result.stdout.strip()), os.path.realpath(str(bin_dir)))

    def test_check_bin_path_script_rejects_empty_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bin_dir = temp_path / "bin"
            bin_dir.mkdir()
            (bin_dir / "bishengir-compile").write_text("", encoding="utf-8")
            (bin_dir / "bishengir-opt").write_text("", encoding="utf-8")
            (temp_path / "config.yaml").write_text(
                f"""bin:
  current: {bin_dir}
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "/Users/ayame/workspace/OJ/check_bin_path.py",
                    "--config",
                    str(temp_path / "config.yaml"),
                    "--key",
                    "current",
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-executable tool binary", result.stderr)

    def test_validate_bin_config_tools_rejects_illegal_tools_before_testcases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            (bin_dir / "bishengir-compile").write_text("", encoding="utf-8")
            (bin_dir / "bishengir-opt").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Non-executable tool binary"):
                validate_bin_config_tools({"current": str(bin_dir)})

    def test_extract_testcase_metrics_reads_required_fields_from_raw_result(self):
        raw_result = {
            "precision_passed": True,
            "eager_time": 100,
            "compile_time": "50.0",
            "current_time": 40.0,
            "max_diff": None,
            "speedup": 2.5,
            "compile_out": object(),
        }

        metrics = extract_testcase_metrics(raw_result)

        self.assertEqual(
            metrics,
            {
                "functional_passed": True,
                "eager_time": 100.0,
                "compile_time": 50.0,
                "current_time": 40.0,
            },
        )

    def test_extract_testcase_metrics_handles_missing_raw_result(self):
        metrics = extract_testcase_metrics(None)

        self.assertFalse(metrics["functional_passed"])
        self.assertIsNone(metrics["eager_time"])
        self.assertIsNone(metrics["compile_time"])
        self.assertIsNone(metrics["current_time"])

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
                "raw_result": {
                    "precision_passed": True,
                    "eager_time": 100.0,
                    "compile_time": 50.0,
                    "current_time": 40.0,
                },
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "raw_result": {
                    "precision_passed": True,
                    "eager_time": 200.0,
                    "compile_time": 100.0,
                    "current_time": 50.0,
                },
            },
            {
                "testcase": "case3.py",
                "exit_code": 1,
                "raw_result": None,
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
                "raw_result": {
                    "precision_passed": True,
                    "eager_time": 100.0,
                    "compile_time": 50.0,
                    "current_time": 40.0,
                },
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "raw_result": {
                    "precision_passed": True,
                    "eager_time": 300.0,
                    "compile_time": 100.0,
                    "current_time": 50.0,
                },
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
                "raw_result": None,
            },
            {
                "testcase": "case2.py",
                "exit_code": 0,
                "raw_result": {
                    "precision_passed": False,
                    "eager_time": None,
                    "compile_time": None,
                    "current_time": None,
                },
            },
        ]
        baseline_data = {}
        now = datetime(2026, 4, 16, 12, 0, 0)

        final_result = generate_final_result(testcase_results, baseline_data, now=now)

        self.assertEqual(final_result["detail"]["functional_score"], 0.0)
        self.assertEqual(final_result["detail"]["performance_score"], 0.0)
        self.assertEqual(final_result["rank"]["rank"], 0.0)

    def test_generate_final_result_marks_functional_failure_as_wa_even_if_exit_code_is_zero(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 0,
                "raw_result": {
                    "precision_passed": False,
                    "eager_time": 100.0,
                    "compile_time": 50.0,
                    "current_time": 40.0,
                },
            }
        ]

        final_result = generate_final_result(testcase_results, {}, now=datetime(2026, 4, 16, 12, 0, 0))

        self.assertEqual(final_result["verdict"], "WA")
        self.assertEqual(final_result["detail"]["passed_testcases"], 0)
        self.assertEqual(final_result["detail"]["failed_testcases"], 1)

    def test_generate_final_result_includes_error_message(self):
        testcase_results = [
            {
                "testcase": "case1.py",
                "exit_code": 1,
                "raw_result": None,
                "error_message": "build_testcase() returned None",
            }
        ]

        final_result = generate_final_result(testcase_results, {}, now=datetime(2026, 4, 16, 12, 0, 0))

        self.assertEqual(final_result["detail"]["testcase_details"][0]["error_message"], "build_testcase() returned None")

    def test_validate_final_result_metrics_accepts_valid_times(self):
        final_result = {
            "detail": {
                "testcase_details": [
                    {
                        "testcase": "case1.py",
                        "exit_code": 0,
                        "eager_time": 100.0,
                        "compile_time": 50.0,
                        "current_time": 40.0,
                    }
                ]
            }
        }

        self.assertIs(validate_final_result_metrics(final_result), final_result)

    def test_validate_final_result_metrics_rejects_missing_profiler_time(self):
        final_result = {
            "detail": {
                "testcase_details": [
                    {
                        "testcase": "case1.py",
                        "exit_code": 0,
                        "eager_time": 100.0,
                        "compile_time": None,
                        "current_time": 40.0,
                    }
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "Please retry the evaluation externally"):
            validate_final_result_metrics(final_result)

    def test_run_testcase_builds_spec_and_returns_raw_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            testcase_path = Path(temp_dir) / "temp_case.py"
            testcase_path.write_text(
                """def build_testcase():
    return {
        "model_or_func": "demo",
        "inputs": (1, 2),
    }
""",
                encoding="utf-8",
            )

            benchmark_calls = []

            def fake_benchmark_runner(**kwargs):
                benchmark_calls.append(kwargs)
                return {
                    "precision_passed": True,
                    "eager_time": 100.0,
                    "compile_time": 50.0,
                    "current_time": 40.0,
                }

            result = run_testcase(str(testcase_path), benchmark_runner=fake_benchmark_runner)

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            result["raw_result"],
            {
                "precision_passed": True,
                "eager_time": 100.0,
                "compile_time": 50.0,
                "current_time": 40.0,
            },
        )
        self.assertEqual(result["error_message"], "")
        self.assertEqual(len(benchmark_calls), 1)
        self.assertEqual(benchmark_calls[0]["model_or_func"], "demo")
        self.assertEqual(benchmark_calls[0]["inputs"], (1, 2))
        self.assertEqual(benchmark_calls[0]["device"], "npu")
        self.assertEqual(benchmark_calls[0]["warmup_steps"], 10)
        self.assertEqual(benchmark_calls[0]["exec_steps"], 30)
        self.assertEqual(benchmark_calls[0]["artifact_subdir"], "temp_case")

    def test_run_testcase_passes_bin_config_to_benchmark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            testcase_path = Path(temp_dir) / "temp_case.py"
            testcase_path.write_text(
                """def build_testcase():
    return {
        "model_or_func": "demo",
        "inputs": (),
    }
""",
                encoding="utf-8",
            )

            benchmark_calls = []

            def fake_benchmark_runner(**kwargs):
                benchmark_calls.append(kwargs)
                return {
                    "precision_passed": True,
                    "eager_time": 100.0,
                    "compile_time": 50.0,
                    "current_time": 40.0,
                }

            result = run_testcase(
                str(testcase_path),
                benchmark_runner=fake_benchmark_runner,
                bin_config={"baseline": "/opt/base/bin", "current": "/opt/current/bin"},
            )

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            benchmark_calls[0]["bin_config"],
            {"baseline": "/opt/base/bin", "current": "/opt/current/bin"},
        )

    def test_run_testcase_wraps_exceptions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            testcase_path = Path(temp_dir) / "temp_case.py"
            testcase_path.write_text(
                """def build_testcase():
    return {
        "model_or_func": "demo",
        "inputs": (),
    }
""",
                encoding="utf-8",
            )

            def fake_benchmark_runner(**kwargs):
                raise RuntimeError("boom")

            result = run_testcase(str(testcase_path), benchmark_runner=fake_benchmark_runner)

        self.assertEqual(result["exit_code"], 1)
        self.assertIsNone(result["raw_result"])
        self.assertEqual(result["error_message"], "boom")

    def test_run_testcase_reports_missing_testcase_spec_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            testcase_path = Path(temp_dir) / "temp_case.py"
            testcase_path.write_text(
                """def build_testcase():
    spec = {
        "model_or_func": "demo",
        "inputs": (),
    }
""",
                encoding="utf-8",
            )

            result = run_testcase(str(testcase_path), benchmark_runner=lambda **kwargs: {})

        self.assertEqual(result["exit_code"], 1)
        self.assertIsNone(result["raw_result"])
        self.assertIn("build_testcase() returned None", result["error_message"])

    def test_run_testcase_reports_missing_benchmark_fields_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            testcase_path = Path(temp_dir) / "temp_case.py"
            testcase_path.write_text(
                """def build_testcase():
    return {
        "model_or_func": "demo",
        "inputs": (),
    }
""",
                encoding="utf-8",
            )

            result = run_testcase(str(testcase_path), benchmark_runner=lambda **kwargs: {"precision_passed": True})

        self.assertEqual(result["exit_code"], 1)
        self.assertIsNone(result["raw_result"])
        self.assertIn("missing keys: compile_time, current_time, eager_time", result["error_message"])

    def test_run_testcase_times_out(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            testcase_path = Path(temp_dir) / "temp_case.py"
            testcase_path.write_text(
                """def build_testcase():
    return {
        "model_or_func": "demo",
        "inputs": (),
    }
""",
                encoding="utf-8",
            )

            def fake_benchmark_runner(**kwargs):
                import time
                time.sleep(2)
                return {
                    "precision_passed": True,
                    "eager_time": 100.0,
                    "compile_time": 50.0,
                    "current_time": 40.0,
                }

            result = run_testcase(str(testcase_path), timeout_seconds=1, benchmark_runner=fake_benchmark_runner)

        self.assertEqual(result["exit_code"], -1)
        self.assertIsNone(result["raw_result"])
        self.assertEqual(result["error_message"], "Timeout expired after 1 seconds")

    def test_build_empty_result_uses_expected_shape(self):
        now = datetime(2026, 4, 16, 12, 0, 0)

        result = build_empty_result(now=now)

        self.assertEqual(result["verdict"], "WA")
        self.assertEqual(result["rank"]["rank"], 0.0)
        self.assertEqual(result["detail"]["timestamp"], now.isoformat())
        self.assertEqual(result["detail"]["testcase_details"], [])


class ProfilerTests(unittest.TestCase):
    def test_resolve_output_dir_appends_subdir(self):
        self.assertEqual(resolve_output_dir("/tmp/prof", "test0"), "/tmp/prof/test0")

    def test_resolve_output_dir_returns_base_dir_for_empty_subdir(self):
        self.assertEqual(resolve_output_dir("/tmp/prof", " . "), "/tmp/prof")

    def test_resolve_output_dir_rejects_outside_path(self):
        with self.assertRaises(ValueError):
            resolve_output_dir("/tmp/prof", "../test0")

    def test_setup_profiler_output_creates_directory_under_subdir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = setup_profiler_output(temp_dir, "test0", print_log=False)
            self.assertTrue(output_dir.startswith(os.path.join(temp_dir, "test0")))
            self.assertTrue(os.path.isdir(output_dir))

    def test_cleanup_directories_removes_single_directory_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "traced_graph_cache"
            (cache_dir / "subdir").mkdir(parents=True)
            (cache_dir / "subdir" / "artifact.txt").write_text("demo", encoding="utf-8")

            removed_count = cleanup_directories([str(cache_dir)], print_log=False)

            self.assertEqual(removed_count, 1)
            self.assertFalse(cache_dir.exists())

    def test_cleanup_directories_removes_prof_and_traced_graph_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prof_dir = Path(temp_dir) / "prof"
            cache_dir = Path(temp_dir) / "traced_graph_cache"
            prof_dir.mkdir()
            cache_dir.mkdir()

            removed_count = cleanup_directories(
                [str(prof_dir), str(cache_dir)],
                print_log=False,
            )

            self.assertEqual(removed_count, 2)
            self.assertFalse(prof_dir.exists())
            self.assertFalse(cache_dir.exists())

    def test_cleanup_directories_skips_missing_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prof_dir = Path(temp_dir) / "prof"
            cache_dir = Path(temp_dir) / "traced_graph_cache"
            prof_dir.mkdir()

            removed_count = cleanup_directories(
                [str(prof_dir), str(cache_dir)],
                print_log=False,
            )

            self.assertEqual(removed_count, 1)
            self.assertFalse(prof_dir.exists())
            self.assertFalse(cache_dir.exists())

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

    def test_get_step_time_from_csv_filters_outliers_for_large_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            values = [100.0] * 20 + [1000.0] * 10
            rows = "\n".join(f"col1,col2,{value},col4,col5" for value in values)
            csv_path.write_text(
                f"header1,header2,header3,header4,header5\n{rows}\n",
                encoding="utf-8",
            )

            result = get_step_time_from_csv(str(csv_path))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result, 100.0)

    def test_get_step_time_from_csv_returns_none_when_no_valid_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            csv_path.write_text("""header1,header2,header3,header4,header5
col1,col2
col1,col2,abc,col4,col5
""", encoding="utf-8")
            
            self.assertIsNone(get_step_time_from_csv(str(csv_path)))



class BinaryManagerTests(unittest.TestCase):
    def setUp(self):
        self._temp_dirs = []
        self._original_path = os.environ.get("PATH", "")

    def tearDown(self):
        os.environ["PATH"] = self._original_path
        for d in self._temp_dirs:
            if os.path.isdir(d):
                import shutil
                shutil.rmtree(d, ignore_errors=True)

    def _make_temp_binary_dir(self):
        """Create a temporary directory to serve as binary/ and register it for cleanup."""
        d = tempfile.mkdtemp()
        self._temp_dirs.append(d)
        return d

    def _make_source_dir_with_tools(self):
        """Create a temp source directory with valid fake tools."""
        src_dir = Path(tempfile.mkdtemp())
        self._temp_dirs.append(str(src_dir))
        write_fake_tool(src_dir / "bishengir-compile")
        write_fake_tool(src_dir / "bishengir-opt")
        return str(src_dir)

    def test_copy_tools_creates_binary_dir(self):
        binary_dir = self._make_temp_binary_dir()
        source_dir = self._make_source_dir_with_tools()

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            os.environ["PATH"] = f"{binary_dir}{os.pathsep}{self._original_path}"
            copy_tools_to_binary(source_dir)

        self.assertTrue(os.path.isfile(os.path.join(binary_dir, "bishengir-compile")))
        self.assertTrue(os.path.isfile(os.path.join(binary_dir, "bishengir-opt")))

    def test_copy_tools_clears_previous(self):
        binary_dir = self._make_temp_binary_dir()
        stale_file = os.path.join(binary_dir, "stale.txt")
        os.makedirs(binary_dir, exist_ok=True)
        Path(stale_file).write_text("old", encoding="utf-8")

        source_dir = self._make_source_dir_with_tools()

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            os.environ["PATH"] = f"{binary_dir}{os.pathsep}{self._original_path}"
            copy_tools_to_binary(source_dir)

        self.assertFalse(os.path.exists(stale_file))
        self.assertTrue(os.path.isfile(os.path.join(binary_dir, "bishengir-compile")))

    def test_copy_tools_rejects_missing_source_dir(self):
        binary_dir = self._make_temp_binary_dir()

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            with self.assertRaises(ToolValidationError):
                copy_tools_to_binary(os.path.join(binary_dir, "nonexistent"))

    def test_copy_tools_rejects_missing_tool(self):
        binary_dir = self._make_temp_binary_dir()
        source_dir = Path(tempfile.mkdtemp())
        self._temp_dirs.append(str(source_dir))
        write_fake_tool(source_dir / "bishengir-compile")
        # bishengir-opt is intentionally missing

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            os.environ["PATH"] = f"{binary_dir}{os.pathsep}{self._original_path}"
            with self.assertRaises(ToolValidationError):
                copy_tools_to_binary(str(source_dir))

    def test_context_does_nothing_for_none(self):
        binary_dir = self._make_temp_binary_dir()

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            with tool_binary_context(None):
                pass

        self.assertEqual(os.listdir(binary_dir), [])

    def test_context_copies_and_verifies(self):
        binary_dir = self._make_temp_binary_dir()
        source_dir = self._make_source_dir_with_tools()

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            os.environ["PATH"] = f"{binary_dir}{os.pathsep}{self._original_path}"
            with tool_binary_context(source_dir):
                self.assertTrue(os.path.isfile(os.path.join(binary_dir, "bishengir-compile")))
                self.assertTrue(os.path.isfile(os.path.join(binary_dir, "bishengir-opt")))

    def test_verify_rejects_wrong_path(self):
        binary_dir = self._make_temp_binary_dir()
        os.makedirs(binary_dir, exist_ok=True)
        # Do NOT put binary/ on PATH, so shutil.which cannot find the tools

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            # Clear PATH so tools are definitely not found
            with unittest.mock.patch.dict(os.environ, {"PATH": ""}, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    _verify_binary_resolves()
                self.assertIn("not found on PATH", str(ctx.exception))

    def test_copy_tools_rejects_empty_tool_file(self):
        binary_dir = self._make_temp_binary_dir()
        source_dir = Path(tempfile.mkdtemp())
        self._temp_dirs.append(str(source_dir))
        # Create empty (too small) tool files
        (source_dir / "bishengir-compile").write_text("", encoding="utf-8")
        (source_dir / "bishengir-opt").write_text("", encoding="utf-8")

        with patch("utils.binary_manager._get_binary_dir", return_value=binary_dir):
            with self.assertRaises(IllegalToolBinaryError):
                copy_tools_to_binary(str(source_dir))


if __name__ == "__main__":
    unittest.main()
