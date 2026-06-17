import logging
import os
import subprocess
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

REQUIRED_TOOLS: Tuple[str, ...] = ("bishengir-compile", "bishengir-opt")
MIN_TOOL_SIZE_BYTES = 10 * 1024 * 1024
VERSION_TIMEOUT_SECONDS = 15


class ToolValidationError(RuntimeError):
    """Raised when a configured bishengir tool cannot be accepted for judging."""


class IllegalToolBinaryError(ToolValidationError):
    """Raised when a submitted bishengir tool is not a valid binary."""


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
        raise ToolValidationError(f"Required tool not found: {resolved_tool_path}")

    file_size = os.stat(resolved_tool_path).st_size
    if file_size <= min_size_bytes:
        raise IllegalToolBinaryError("Illegal bishengir tool binary")

    if not os.access(resolved_tool_path, os.X_OK):
        raise IllegalToolBinaryError("Illegal bishengir tool binary")

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
        raise IllegalToolBinaryError("Illegal bishengir tool binary") from exc
    except OSError as exc:
        raise IllegalToolBinaryError("Illegal bishengir tool binary") from exc

    if result.returncode != 0:
        raise IllegalToolBinaryError("Illegal bishengir tool binary")

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
        raise ToolValidationError(f"Configured bin.{key} directory does not exist: {resolved_bin_dir}")

    # for tool_name in tool_names:
    #     validate_tool_file(
    #         os.path.join(resolved_bin_dir, tool_name),
    #         min_size_bytes=min_size_bytes,
    #         version_timeout_seconds=version_timeout_seconds,
    #     )

    return resolved_bin_dir
