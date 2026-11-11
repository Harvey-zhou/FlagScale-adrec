#!/bin/bash
# run-resumable.sh <card> <name> <ads|rec> <config> [extra overrides...]
#
# Runs a job and relaunches it after a crash, resuming from the per-epoch
# `latest.pt` that train_ads/train_rec write.
#
# Two failure modes are handled:
#   * the vendor runtime dies with SIGABRT ("wait for noc idle timeout") after
#     tens of hours of continuous training;
#   * a degraded card makes the *preflight* matmul hang, so the launcher never
#     returns.  A plain wait-for-exit loop blocks forever there, so the attempt
#     is additionally killed when history.json has not advanced for STALL_S
#     seconds.
# The launcher reports exit 0 even on a crash, so success is keyed on
# metrics.json instead of the exit code.
set -uo pipefail
card=$1; shift
name=$1; shift
module=$1; shift
config=$1; shift

root=/data2/jiangxuhui/FlagScale-adrec-harvey-zhou
out=/data2/jiangxuhui/logs/adrec-runs/$name
STALL_S=${STALL_S:-2700}   # 45 min without a new epoch => kill the attempt
mkdir -p "$out" "/data2/jiangxuhui/cache/jitc-temp-$name" "/data2/jiangxuhui/cache/xblas-$name"
source /data2/jiangxuhui/logs/adrec-runs/env.sh
export JITC_TEMP_DIR="/data2/jiangxuhui/cache/jitc-temp-$name"
export XBLAS_JITC_CACHE_PATH="/data2/jiangxuhui/cache/xblas-$name"
export ADREC_OUT_ROOT="$out"

cd "$root" || exit 1
attempt=0
while [ "$attempt" -lt 40 ]; do
  attempt=$((attempt + 1))
  if [ -n "$(find "$out" -name metrics.json -print -quit 2>/dev/null)" ]; then
    echo "===== $name: metrics.json present, done ====="
    exit 0
  fi
  resume=1
  [ -z "$(find "$out" -name latest.pt -print -quit 2>/dev/null)" ] && resume=0
  echo "===== $name attempt $attempt (resume=$resume) $(date '+%F %T') ====="

  # a crash can leave the worker alive as an orphan; the resumed run must own
  # the checkpoint directory exclusively
  for pattern in "train_rec.py --config-file=$out" "train_ads.py --config-file=$out"; do
    pgrep -f "$pattern" 2>/dev/null | while read -r pid; do kill -9 "$pid" 2>/dev/null; done
  done
  sleep 5

  FLAGSCALE_RESUME=$resume bash examples/run-adrec-kunlunxin.sh "$module" "$config" "$card" "$@" \
    >> "$out/launch.log" 2>&1 &
  launcher=$!
  history=$(find "$out" -name history.json -print -quit 2>/dev/null)
  started_at=$(date +%s)
  last_mark=$started_at
  # a stale history from a previous attempt must not trigger the stall detector
  # immediately: the clock starts when this attempt starts
  if [ -n "$history" ]; then
    mark=$(stat -c %Y "$history")
    [ "$mark" -gt "$last_mark" ] && last_mark=$mark
  fi

  while kill -0 "$launcher" 2>/dev/null; do
    sleep 60
    if [ -n "$(find "$out" -name metrics.json -print -quit 2>/dev/null)" ]; then
      wait "$launcher" 2>/dev/null
      echo "===== $name: metrics.json present, done ====="
      exit 0
    fi
    history=$(find "$out" -name history.json -print -quit 2>/dev/null)
    if [ -n "$history" ]; then
      mark=$(stat -c %Y "$history")
      [ "$mark" -gt "$last_mark" ] && last_mark=$mark
      if [ $(( $(date +%s) - last_mark )) -gt "$STALL_S" ]; then
        echo "===== $name: no epoch for ${STALL_S}s (stuck in preflight or hung), killing attempt $attempt ====="
        pkill -9 -P "$launcher" 2>/dev/null
        kill -9 "$launcher" 2>/dev/null
        for pattern in "train_rec.py --config-file=$out" "train_ads.py --config-file=$out"; do
          pgrep -f "$pattern" 2>/dev/null | while read -r pid; do kill -9 "$pid" 2>/dev/null; done
        done
        break
      fi
    elif [ $(( $(date +%s) - started_at )) -gt "$STALL_S" ]; then
      echo "===== $name: no history after ${STALL_S}s, killing attempt $attempt ====="
      pkill -9 -P "$launcher" 2>/dev/null
      kill -9 "$launcher" 2>/dev/null
      break
    fi
  done
  wait "$launcher" 2>/dev/null
  sleep 60
done
echo "===== $name: gave up after $attempt attempts ====="
exit 1
