#!/usr/bin/env python3

import argparse
import os
import sys

from case_judge import load_bin_config
from utils.tool_validation import validate_tool_bin_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Validate bishengir tool bin paths from config.yaml.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--key",
        choices=("baseline", "current"),
        default="current",
        help="Which configured bin path to validate and print.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bin_config = load_bin_config(args.config)
    if bin_config is None:
        raise ValueError(f"No bin configuration found in {args.config}")

    bin_dir = bin_config.get(args.key)
    if not bin_dir:
        raise ValueError(f"bin.{args.key} is not configured in {args.config}")

    print(validate_tool_bin_dir(bin_dir, args.key))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
