#!/bin/bash
# run-resumable.sh <card> <name> <ads|rec> <config> [extra overrides...]
#
# Runs a job and relaunches it after a crash, resuming from the per-epoch
# `latest.pt` that train_ads/train_rec now write.  The vendor runtime dies with
# SIGABRT ("wait for noc idle timeout") after tens of hours of continuous
# training, and the launcher still reports exit 0, so the loop keys on the
# success marker (metrics.json) instead of the exit code.
set -uo pipefail
card=$1; shift
name=$1; shift
module=$1; shift
config=$1; shift

root=/data2/jiangxuhui/FlagScale-adrec-harvey-zhou
out=/data2/jiangxuhui/logs/adrec-runs/$name
mkdir -p "$out" "/data2/jiangxuhui/cache/jitc-temp-$name" "/data2/jiangxuhui/cache/xblas-$name"
source /data2/jiangxuhui/logs/adrec-runs/env.sh
export JITC_TEMP_DIR="/data2/jiangxuhui/cache/jitc-temp-$name"
export XBLAS_JITC_CACHE_PATH="/data2/jiangxuhui/cache/xblas-$name"
export ADREC_OUT_ROOT="$out"

cd "$root" || exit 1
attempt=0
while [ "$attempt" -lt 40 ]; do
  attempt=$((attempt + 1))
  metrics=$(find "$out" -name metrics.json -print -quit 2>/dev/null)
  if [ -n "$metrics" ]; then
    echo "===== $name: metrics.json present, done ====="
    exit 0
  fi
  resume=1
  if [ -z "$(find "$out" -name latest.pt -print -quit 2>/dev/null)" ]; then
    resume=0
  fi
  echo "===== $name attempt $attempt (resume=$resume) $(date '+%F %T') ====="
  FLAGSCALE_RESUME=$resume bash examples/run-adrec-kunlunxin.sh "$module" "$config" "$card" "$@" \
    >> "$out/launch.log" 2>&1
  sleep 60
done
echo "===== $name: gave up after $attempt attempts ====="
exit 1
