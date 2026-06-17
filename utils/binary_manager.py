"""
Utility module for managing a project-local binary/ directory that holds
copies of bishengir tools (bishengir-compile, bishengir-opt).

Instead of manipulating PATH to point at different source directories,
tools are copied from the configured source into binary/.  The run.sh
script places ./binary on PATH, so subprocesses always find the tools
that were most recently copied.
"""

import logging
import os
import shutil
import stat
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from utils.tool_validation import REQUIRED_TOOLS, validate_tool_file

logger = logging.getLogger(__name__)

BINARY_DIR_NAME = "binary"

_lock = threading.Lock()


def _get_project_root() -> str:
    """Return the absolute path of the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_binary_dir() -> str:
    """Return the absolute path to the local binary/ directory."""
    return os.path.join(_get_project_root(), BINARY_DIR_NAME)


def _ensure_binary_dir() -> str:
    """Create the binary/ directory if it does not exist and return its path."""
    binary_dir = _get_binary_dir()
    os.makedirs(binary_dir, exist_ok=True)
    return binary_dir


def _clear_binary_dir() -> None:
    """Remove all files and directories inside binary/."""
    binary_dir = _get_binary_dir()
    if not os.path.isdir(binary_dir):
        return

    for entry in os.listdir(binary_dir):
        entry_path = os.path.join(binary_dir, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
        except OSError:
            logger.exception("Failed to remove %s while clearing binary/", entry_path)
            raise


def _verify_binary_resolves() -> None:
    """Check that each required tool resolves to the local binary/ directory."""
    binary_dir = _get_binary_dir()

    for tool_name in REQUIRED_TOOLS:
        resolved = shutil.which(tool_name)
        if resolved is None:
            raise RuntimeError(
                f"Tool {tool_name!r} not found on PATH after copy. "
                f"Ensure {binary_dir} is on PATH."
            )

        resolved_real = os.path.realpath(resolved)
        expected = os.path.join(binary_dir, tool_name)
        expected_real = os.path.realpath(expected)

        if resolved_real != expected_real:
            raise RuntimeError(
                f"Tool {tool_name!r} resolves to {resolved_real!r}, "
                f"expected {expected_real!r}. "
                f"Check PATH ordering."
            )

    logger.info("Verified all tools resolve to local binary directory: %s", binary_dir)


def copy_tools_to_binary(source_dir: str) -> None:
    """Clear binary/, copy tools from *source_dir*, and verify PATH resolution.

    Each tool is validated via :func:`validate_tool_file` before copying.
    After the copy the execute permission is explicitly set on every tool.
    """
    binary_dir = _ensure_binary_dir()
    _clear_binary_dir()

    for tool_name in REQUIRED_TOOLS:
        source_path = os.path.join(source_dir, tool_name)

        # Validate before copying — raises ToolValidationError or subclass on failure
        validate_tool_file(source_path)

        dest_path = os.path.join(binary_dir, tool_name)
        shutil.copy2(source_path, dest_path)

        # Ensure execute bits are set regardless of copy2 behaviour
        current_mode = os.stat(dest_path).st_mode
        os.chmod(
            dest_path,
            current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )

        logger.info("Copied %s -> %s", source_path, dest_path)

    _verify_binary_resolves()


@contextmanager
def tool_binary_context(source_dir: Optional[str]) -> Iterator[None]:
    """Context manager that swaps the tools in binary/ to those from *source_dir*.

    When *source_dir* is ``None`` or an empty string this is a no-op.

    The context manager acquires a lock while copying, so only one thread
    can mutate binary/ at a time.  The lock is released before yielding
    to the caller, allowing concurrent tool execution.
    """
    if not source_dir:
        yield
        return

    _lock.acquire()
    try:
        copy_tools_to_binary(source_dir)
    finally:
        _lock.release()

    try:
        yield
    finally:
        pass  # no cleanup — next invocation will clear-and-recopy
