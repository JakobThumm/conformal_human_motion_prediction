#!/bin/bash
# Wait until the shared GPU is free, then launch the full coverage sweep. Polls nvidia-smi every
# POLL seconds; when used GPU memory stays below THRESHOLD_MIB for STABLE consecutive checks (so we
# don't pounce on a momentary dip), runs run_full_sweep.sh. Caps total wait at MAX_HOURS.
cd "$(dirname "$0")/../.."

THRESHOLD=${THRESHOLD_MIB:-6000}     # consider the GPU "free" below this many MiB used
POLL=${POLL:-30}
STABLE=${STABLE:-3}
MAX_HOURS=${MAX_HOURS:-48}
SWEEP_LOG=/tmp/coverage_sweep.log

iters=$(( MAX_HOURS * 3600 / POLL ))
echo "[watcher] pid=$$ started $(date) | threshold=${THRESHOLD}MiB poll=${POLL}s stable=${STABLE} max=${MAX_HOURS}h"

ok=0
for i in $(seq 1 "$iters"); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$used" ] && [ "$used" -lt "$THRESHOLD" ]; then
    ok=$((ok+1))
  else
    ok=0
  fi
  if [ $(( i % 20 )) -eq 1 ]; then echo "[watcher] $(date) used=${used:-?}MiB ok_streak=${ok}/${STABLE}"; fi
  if [ "$ok" -ge "$STABLE" ]; then
    echo "[watcher] GPU free ($(date), used=${used}MiB). Launching sweep -> ${SWEEP_LOG}"
    bash experiments/coverage/run_full_sweep.sh > "$SWEEP_LOG" 2>&1
    rc=$?
    echo "[watcher] sweep finished rc=${rc} $(date)"
    exit $rc
  fi
  sleep "$POLL"
done
echo "[watcher] timed out after ${MAX_HOURS}h waiting for GPU $(date)"
exit 1
