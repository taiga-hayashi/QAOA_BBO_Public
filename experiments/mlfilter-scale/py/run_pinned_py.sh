#!/usr/bin/env bash
# Run a Python script of this folder pinned to one free core (flock-held for the job's lifetime).
# usage: [CORE_LIST="4 5 6 7"] run_pinned_py.sh <script.py> [args...]   (default cores 8-47)
set -euo pipefail
PY=/home/jovyan/FMA/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOCKS=/home/jovyan/FMA/Fujitsu_OpenQarp/cache/core-locks
mkdir -p "$LOCKS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
script=$1; shift
for core in ${CORE_LIST:-$(seq 8 47)}; do
  exec 9>"$LOCKS/core$core.lock"
  if flock -n 9; then
    echo "core=$core host=$(hostname) start=$(date -u +%FT%TZ) script=$script args=$*"
    status=0
    taskset -c "$core" "$PY" -u "$HERE/$script" "$@" || status=$?
    echo "end=$(date -u +%FT%TZ) exit=$status"
    exit $status
  fi
  exec 9>&-
done
echo "no free core" >&2
exit 75
