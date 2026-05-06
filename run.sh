#!/bin/bash

# check bishengir-compile and bishengir-opt exist
curdir="/coursegrader/submit"
if [ ! -e "$curdir/bishengir-compile" || ! -e "$curdir/bishengir-opt" ]; then
    echo "please put bishengir-compile and bishengir-opt in $curdir"
    exit 1
fi

# add execute permission
chmod +x $curdir
export PATH=$curdir:$PATH
which bishengir-compile
which bishengir-opt

# run case_judge.py
python3 ./case_judge.py "$@"
