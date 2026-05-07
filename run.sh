#!/bin/bash

curdir=$(python3 ./check_bin_path.py --key current) || exit 1

export PATH=$curdir:$PATH
which bishengir-compile
which bishengir-opt

# run case_judge.py
python3 ./case_judge.py "$@"
