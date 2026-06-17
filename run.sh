#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

: "${LOG_LEVEL:=CRITICAL}"
export LOG_LEVEL

if [[ "${LOG_LEVEL^^}" == "DEBUG" ]]; then
    : "${OJ_RUN_LOG:=/dev/stdout}"
else
    : "${OJ_RUN_LOG:=/dev/null}"
fi
exec 2>"$OJ_RUN_LOG"

git pull --quiet >&2 || true

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh >&2 || true
fi

echo "BISHENG_INSTALL_PATH: $BISHENG_INSTALL_PATH" >&2

export PATH="$PWD/binary:$PATH"

python_status=0
python_output=$(python3 ./case_judge.py "$@") || python_status=$?
if [ -n "$python_output" ]; then
    python_output=${python_output%$'\n'}
    printf '%s\n' "${python_output##*$'\n'}"
else
    printf '{"verdict":"CE","rank":{"rank":-1},"score":0,"comment":"case_judge.py exited without producing a result","detail":"exit_code=%s"}\n' "$python_status"
fi
exit "$python_status"
