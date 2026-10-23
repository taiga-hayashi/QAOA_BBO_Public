#!/usr/bin/env bash
# Run one phase0.py subcommand pinned to free cores (flock-held for the job's lifetime).
# usage: [NCORES=n] run_pinned.sh <subcommand> <run_dir>
# NCORES (default 1) cores are locked and pinned; OMP_NUM_THREADS=NCORES is used by the qarpx simulator.
set -euo pipefail
PY=/home/jovyan/FMA/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
LOCKS=/home/jovyan/FMA/Fujitsu_OpenQarp/cache/core-locks
NCORES=${NCORES:-1}
mkdir -p "$LOCKS"
cores=(); fd=10
for core in $(seq 8 47); do
  [ ${#cores[@]} -ge "$NCORES" ] && break
  eval "exec $fd>\"$LOCKS/core$core.lock\""
  if flock -n $fd; then cores+=("$core"); fd=$((fd+1)); else eval "exec $fd>&-"; fi
done
if [ ${#cores[@]} -lt "$NCORES" ]; then echo "only ${#cores[@]} free cores, need $NCORES" >&2; exit 75; fi
list=$(IFS=,; echo "${cores[*]}")
export OMP_NUM_THREADS=$NCORES OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "cores=$list omp_threads=$NCORES host=$(hostname) start=$(date -u +%FT%TZ) cmd=$1 run=$2"
status=0
taskset -c "$list" "$PY" -u "$HERE/phase0.py" "$1" --run-dir "$2" || status=$?
echo "end=$(date -u +%FT%TZ) exit=$status"
exit $status
