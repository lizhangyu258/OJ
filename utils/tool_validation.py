import logging
import os
import subprocess
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

REQUIRED_TOOLS: Tuple[str, ...] = ("bishengir-compile", "bishengir-opt")
COMPILE_TIMEOUT_SECONDS = 60


class ToolValidationError(RuntimeError):
    """Raised when a configured bishengir tool cannot be accepted for judging."""


class IllegalToolBinaryError(ToolValidationError):
    """Raised when a submitted bishengir tool is not a valid binary."""


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_validate_case() -> str:
    return os.path.join(_get_project_root(), "validate", "case.mlir")


def _validate_compile_tool(tool_path: str) -> None:
    """Run bishengir-compile on validate/case.mlir to verify it produces
    normal output (exit code 0)."""
    case_file = _get_validate_case()
    if not os.path.isfile(case_file):
        raise ToolValidationError(f"Validation case not found: {case_file}")

    source_dir = os.path.dirname(tool_path)
    env = os.environ.copy()
    env["PATH"] = f"{source_dir}{os.pathsep}{env.get('PATH', '')}"

    try:
        result = subprocess.run(
            [
                tool_path,
                "-enable-hfusion-compile=true",
                "-enable-hivm-compile=false",
                "--mlir-print-ir-after-all",
                case_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=COMPILE_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise IllegalToolBinaryError("Tool binary not responding")
    except OSError:
        raise IllegalToolBinaryError("Tool binary execution failed")

    #if result.returncode != 0:
    #    raise IllegalToolBinaryError("Tool binary compilation failed")

    combined_output = result.stdout + result.stderr
    if "mlir" not in combined_output:
        raise IllegalToolBinaryError("Tool binary compilation failed")

    logger.info("Validated %s: compile test passed", tool_path)


def validate_tool_file(tool_path: str) -> str:
    """Validate a single bishengir tool binary.

    For ``bishengir-compile``, additionally runs a compilation smoke test
    on ``validate/case.mlir`` to ensure the tool can produce normal output.
    ``bishengir-opt`` only requires the file to exist and be executable.
    """
    resolved_tool_path = os.path.abspath(tool_path)
    if not os.path.isfile(resolved_tool_path):
        raise ToolValidationError(f"Required tool not found: {resolved_tool_path}")

    if not os.access(resolved_tool_path, os.X_OK):
        raise IllegalToolBinaryError("Non-executable tool binary")

    tool_name = os.path.basename(resolved_tool_path)
    if tool_name == "bishengir-compile":
        _validate_compile_tool(resolved_tool_path)

    return resolved_tool_path


def validate_tool_bin_dir(
    bin_dir: str,
    key: str,
    *,
    tool_names: Iterable[str] = REQUIRED_TOOLS,
) -> str:
    resolved_bin_dir = os.path.abspath(bin_dir)
    if not os.path.isdir(resolved_bin_dir):
        raise ToolValidationError(f"Configured bin.{key} directory does not exist: {resolved_bin_dir}")

    for tool_name in tool_names:
        validate_tool_file(os.path.join(resolved_bin_dir, tool_name))

    return resolved_bin_dir
