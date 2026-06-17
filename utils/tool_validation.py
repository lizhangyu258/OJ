import logging
import os
import subprocess
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

REQUIRED_TOOLS: Tuple[str, ...] = ("bishengir-compile", "bishengir-opt")
MIN_TOOL_SIZE_BYTES = 10 * 1024 * 1024
VERSION_TIMEOUT_SECONDS = 15


def _short_output(value: str, limit: int = 1000) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def validate_tool_file(
    tool_path: str,
    *,
    min_size_bytes: int = MIN_TOOL_SIZE_BYTES,
    version_timeout_seconds: int = VERSION_TIMEOUT_SECONDS,
) -> str:
    resolved_tool_path = os.path.abspath(tool_path)
    if not os.path.isfile(resolved_tool_path):
        raise FileNotFoundError(f"Required tool not found: {resolved_tool_path}")

    file_size = os.stat(resolved_tool_path).st_size
    if file_size <= min_size_bytes:
        raise ValueError(
            f"Required tool is too small: {resolved_tool_path} "
            f"({file_size} bytes <= {min_size_bytes} bytes)"
        )

    if not os.access(resolved_tool_path, os.X_OK):
        raise PermissionError(f"Required tool is not executable: {resolved_tool_path}")

    try:
        result = subprocess.run(
            [resolved_tool_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=version_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Tool version probe timed out after {version_timeout_seconds}s: {resolved_tool_path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to execute tool directly: {resolved_tool_path}: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Tool version probe failed: {resolved_tool_path} "
            f"(exit_code={result.returncode}, "
            f"stdout={_short_output(result.stdout)!r}, "
            f"stderr={_short_output(result.stderr)!r})"
        )

    version_text = _short_output(result.stdout or result.stderr)
    if version_text:
        logger.info("Validated %s: %s", resolved_tool_path, version_text)
    else:
        logger.info("Validated %s with an empty --version output", resolved_tool_path)
    return resolved_tool_path


def validate_tool_bin_dir(
    bin_dir: str,
    key: str,
    *,
    tool_names: Iterable[str] = REQUIRED_TOOLS,
    min_size_bytes: int = MIN_TOOL_SIZE_BYTES,
    version_timeout_seconds: int = VERSION_TIMEOUT_SECONDS,
) -> str:
    resolved_bin_dir = os.path.abspath(bin_dir)
    if not os.path.isdir(resolved_bin_dir):
        raise FileNotFoundError(f"Configured bin.{key} directory does not exist: {resolved_bin_dir}")

    for tool_name in tool_names:
        validate_tool_file(
            os.path.join(resolved_bin_dir, tool_name),
            min_size_bytes=min_size_bytes,
            version_timeout_seconds=version_timeout_seconds,
        )

    return resolved_bin_dir
