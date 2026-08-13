#!/usr/bin/env python3
"""Run bishengir-compile while preserving its pass-manager IR output.

When IR caching is enabled, binary_manager installs this file as
``binary/bishengir-compile`` and keeps the submitted compiler beside it as
``bishengir-compile.real``.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


REAL_COMPILER_NAME = "bishengir-compile.real"
IR_DUMP_DIR_ENV = "OJ_BISHENGIR_IR_DUMP_DIR"
IR_PRINT_OPTION = "--mlir-print-ir-after-all"
MAX_IR_DUMP_BYTES_ENV = "OJ_BISHENGIR_IR_MAX_BYTES"
DEFAULT_MAX_IR_DUMP_BYTES = 512 * 1024 * 1024


def _with_ir_printing(arguments):
    """Enable pass-manager IR printing exactly once."""
    result = []
    found = False
    for argument in arguments:
        if argument == IR_PRINT_OPTION or argument.startswith(
            f"{IR_PRINT_OPTION}="
        ):
            if not found:
                result.append(IR_PRINT_OPTION)
                found = True
            continue
        result.append(argument)
    if not found:
        # Keep the option before a possible ``--`` downstream-argument
        # separator.  Appending after it would forward the option to hivmc
        # instead of enabling the bishengir-compile pass manager.
        try:
            separator_index = result.index("--")
        except ValueError:
            separator_index = 0
        result.insert(separator_index, IR_PRINT_OPTION)
    return result


def _is_metadata_only_invocation(arguments):
    """Do not add compilation flags to --version/--help probes."""
    return any(
        argument in ("--version", "-version", "--help", "-help")
        for argument in arguments
    )


def _input_stem(arguments):
    for argument in arguments:
        if argument.endswith(".mlir"):
            return Path(argument).stem
    return "unknown-input"


def _unique_prefix(dump_dir, arguments):
    token = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return Path(dump_dir) / f"{token}-{_input_stem(arguments)}"


def _write_metadata(path, real_compiler, original_arguments, arguments):
    metadata = {
        "cwd": os.getcwd(),
        "compiler": str(real_compiler),
        "original_arguments": original_arguments,
        "effective_arguments": arguments,
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _max_dump_bytes():
    value = os.environ.get(MAX_IR_DUMP_BYTES_ENV)
    if value is None:
        return DEFAULT_MAX_IR_DUMP_BYTES
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_IR_DUMP_BYTES
    return max(0, parsed)


def main():
    wrapper_dir = Path(__file__).resolve().parent
    real_compiler = wrapper_dir / REAL_COMPILER_NAME
    dump_dir = os.environ.get(IR_DUMP_DIR_ENV)

    if not dump_dir:
        os.execv(str(real_compiler), [str(real_compiler), *sys.argv[1:]])

    if _is_metadata_only_invocation(sys.argv[1:]):
        os.execv(str(real_compiler), [str(real_compiler), *sys.argv[1:]])

    original_arguments = sys.argv[1:]
    arguments = _with_ir_printing(original_arguments)
    try:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        prefix = _unique_prefix(dump_dir, arguments)
        _write_metadata(
            Path(f"{prefix}.command.json"),
            real_compiler,
            original_arguments,
            arguments,
        )
        ir_stream = Path(f"{prefix}.ir.log").open("wb")
    except OSError as exc:
        print(
            f"[OJ] Cannot create bishengir IR dump under {dump_dir}: {exc}",
            file=sys.stderr,
        )
        os.execv(str(real_compiler), [str(real_compiler), *arguments])

    try:
        process = subprocess.Popen(
            [str(real_compiler), *arguments],
            stdout=None,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        ir_stream.close()
        print(f"[OJ] Failed to start bishengir-compile: {exc}", file=sys.stderr)
        return 127

    assert process.stderr is not None
    max_dump_bytes = _max_dump_bytes()
    dumped_bytes = 0
    dump_limit_reported = False
    try:
        while True:
            chunk = process.stderr.read(64 * 1024)
            if not chunk:
                break
            if ir_stream is not None:
                try:
                    remaining_bytes = max_dump_bytes - dumped_bytes
                    if remaining_bytes > 0:
                        written_chunk = chunk[:remaining_bytes]
                        ir_stream.write(written_chunk)
                        dumped_bytes += len(written_chunk)
                    if dumped_bytes >= max_dump_bytes:
                        ir_stream.close()
                        ir_stream = None
                        if not dump_limit_reported:
                            print(
                                "[OJ] bishengir IR dump reached the per-invocation "
                                f"limit of {max_dump_bytes} bytes; remaining IR is not cached",
                                file=sys.stderr,
                            )
                            dump_limit_reported = True
                except OSError as exc:
                    print(
                        f"[OJ] Failed to save bishengir IR dump: {exc}",
                        file=sys.stderr,
                    )
                    ir_stream.close()
                    ir_stream = None
            try:
                os.write(sys.stderr.fileno(), chunk)
            except OSError:
                # A closed OJ diagnostics stream must not cause compilation CE.
                pass
        return process.wait()
    finally:
        if ir_stream is not None:
            ir_stream.close()


if __name__ == "__main__":
    sys.exit(main())
