#!/bin/bash
# Run one advertising (ads) or recommendation (rec) model on Cambricon MLU cards.
#
#   bash examples/run-adrec-cambricon.sh ads din 0
#   bash examples/run-adrec-cambricon.sh rec lightgcn_yelp2018 1
#   bash examples/run-adrec-cambricon.sh ads deepfm 0,1                 # two cards, DDP over CNCL
#   bash examples/run-adrec-cambricon.sh rec sasrec 0 train.epochs=1    # extra Hydra overrides
#   PREFLIGHT_ONLY=1 bash examples/run-adrec-cambricon.sh ads din 0     # check the stack, do not train
#   SMOKE=1 bash examples/run-adrec-cambricon.sh ads din_paper 0        # operator probe: 3 steps + 1 eval
#   SMOKE=1 bash examples/run-adrec-cambricon.sh rec lightgcn 0 --propagation index_add
#   SMOKE=1 ALLOW_CPU_FALLBACK=1 bash examples/run-adrec-cambricon.sh ads din 0
#                                   # list, in one pass, every operator torch_mlu would run on the CPU
#
# The stack is the vendor's torch_mlu + CNCL.  torch_mlu claims PyTorch's PrivateUse1 slot as
# `mlu` and flagscale/platforms/platform_mlu.py drives it (`device_type=mlu`, backend `cncl`).
# Torch-FL is NOT involved, the same way run-ascend.sh keeps it out of the torch_npu path.
#
# CPU fallback.  torch_mlu runs an operator it has no MLU kernel for on the CPU and only warns
# once per operator: ENABLE_FALLBACK_TO_CPU is on when unset (csrc/aten/MLUFallback.cpp, read
# on torch_mlu r1.22_pt2.1.0).  That is a silent fallback, so this script sets
# ENABLE_FALLBACK_TO_CPU=0 and an unsupported operator fails the run.  As a second guard that
# does not depend on the variable's name in the installed version, the log is scanned for
# torch_mlu's "will fall back to run on the CPU" warning, and a hit fails the run.
# ALLOW_CPU_FALLBACK=1 lifts both, for the operator probe only; such a run is not a result.
#
# LightGCN: whether torch.sparse.mm runs on the MLU is not known yet, so nothing is forced.
# Probe it with SMOKE=1; if it fails, train with train.model.propagation=index_add, which is
# mathematically identical (the same switch Torch-FL and Kunlunxin use).
#
# Environment (cluster-specific defaults are marked TODO(cluster); fill them in on this branch):
#   CAMBRICON_ENV_SH  optional: sourced first, for a host that activates the stack its own way
#   ADREC_VENV        the venv/conda env holding torch + torch_mlu; tried in order: this
#                     variable, <repo>/venv, then the python already on PATH
#   NEUWARE_HOME      Cambricon SDK root (default /usr/local/neuware when it exists)
#   ADREC_DATA_ROOT   datasets root (default /flagos-ads/datasets or /flagos-rec/datasets)
#   ADREC_OUT_ROOT    checkpoints, logs and scratch (default $HOME/flagos-adrec-cambricon); keep
#                     it on a disk with free space
#   OMP_NUM_THREADS   default 1.  The container's cgroup may grant only a few CPU cores, and the
#                     per-model configs also ask for dataloader workers.
#   ACTION            FlagScale action (default test, which runs in the foreground)
set -uo pipefail

module=${1:?usage: run-adrec-cambricon.sh <ads|rec> <config> [cards] [hydra overrides...]}
config=${2:?config name, e.g. din, deepfm, lightgcn_yelp2018, sasrec_beauty}
cards=${3:-0}
extra=("${@:4}")
case "$module" in
  ads) data_var=FLAGOS_ADS_DATA; ckpt_var=FLAGOS_ADS_CKPTS; data_default=/flagos-ads/datasets ;;
  rec) data_var=FLAGOS_REC_DATA; ckpt_var=FLAGOS_REC_CKPTS; data_default=/flagos-rec/datasets ;;
  *) echo "module must be ads or rec, got '$module'" >&2; exit 2 ;;
esac
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
if [ ! -f "$ROOT/examples/$module/conf/train/$config.yaml" ]; then
  echo "no such config: $ROOT/examples/$module/conf/train/$config.yaml" >&2
  exit 2
fi

if [ -n "${CAMBRICON_ENV_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "$CAMBRICON_ENV_SH"
fi

# ---- interpreter -------------------------------------------------------------------------
# TODO(cluster): add the cluster's torch_mlu env as a candidate once its path is known.
VENV=""
for candidate in "${ADREC_VENV:-}" "$ROOT/venv"; do
  if [ -n "$candidate" ] && [ -x "$candidate/bin/python" ]; then VENV=$candidate; break; fi
done
if [ -n "$VENV" ]; then
  PY="$VENV/bin/python"
  export PATH="$VENV/bin:$PATH"
else
  PY=$(command -v python3 || command -v python || true)
  if [ -z "$PY" ]; then
    echo "no python found; set ADREC_VENV to the env that has torch_mlu" >&2
    exit 2
  fi
fi

# ---- vendor stack ------------------------------------------------------------------------
# TODO(cluster): confirm the SDK location; the official containers usually set it already.
if [ -z "${NEUWARE_HOME:-}" ] && [ -d /usr/local/neuware ]; then
  export NEUWARE_HOME=/usr/local/neuware
fi
if [ -n "${NEUWARE_HOME:-}" ]; then
  export PATH="$NEUWARE_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export MLU_VISIBLE_DEVICES="$cards"
if [ "${ALLOW_CPU_FALLBACK:-0}" = 1 ]; then
  export ENABLE_FALLBACK_TO_CPU=1
else
  export ENABLE_FALLBACK_TO_CPU=0
fi
# Pin the platform here, so a FLAGSCALE_DEVICE / FS_PLATFORM left in the shell cannot override
# what conf/train.yaml derives from device_type.
export FLAGSCALE_DEVICE=mlu FS_PLATFORM=mlu FLAGSCALE_DISTRIBUTED_BACKEND=cncl

OUT=${ADREC_OUT_ROOT:-$HOME/flagos-adrec-cambricon}/$module
LOGS=$OUT/logs
export TMPDIR="$OUT/tmp"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

nproc=$(awk -F, '{print NF}' <<< "$cards")
export "$data_var"="${ADREC_DATA_ROOT:-${!data_var:-$data_default}}"
export "$ckpt_var"="$OUT/ckpts"
export FLAGSCALE_HOME="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE=${WANDB_MODE:-offline}

# Names every operator torch_mlu reported running on the CPU, or refusing to run.  Returns 1
# when a fallback happened and ALLOW_CPU_FALLBACK is not set.
scan_log() {
  local log=$1 fallback missing
  fallback=$(grep -o "The operator '[^']*'[^']*fall back to run on the CPU" "$log" \
    | sed "s/^The operator '\([^']*\)'.*/\1/" | sort -u)
  missing=$(grep -o "The operator '[^']*' is not currently implemented for the MLU device" "$log" \
    | sed "s/^The operator '\([^']*\)'.*/\1/" | sort -u)
  if [ -n "$missing" ]; then
    echo "===== operators with no MLU kernel (run stopped) ====="
    echo "$missing"
  fi
  if [ -n "$fallback" ]; then
    echo "===== operators torch_mlu ran on the CPU ====="
    echo "$fallback"
    if [ "${ALLOW_CPU_FALLBACK:-0}" != 1 ]; then
      echo "CPU fallback happened without ALLOW_CPU_FALLBACK=1; this run is not an MLU result" >&2
      return 1
    fi
  fi
  return 0
}

# ---- preflight ---------------------------------------------------------------------------
# Asserts the stack is live and records the versions and capabilities the training code
# depends on, so a gap is named here rather than inferred from a later failure.  The log is
# also the source for the environment table in examples/CAMBRICON.md.
echo "===== preflight: $PY (cards=$cards) ====="
{
  if command -v cnmon >/dev/null 2>&1; then cnmon || true; fi
  "$PY" - "$nproc" <<'PY'
import os
import sys

import torch

try:
    import torch_mlu
except Exception as exc:
    sys.exit(f"torch_mlu does not import in {sys.executable}: {type(exc).__name__}: {exc}")

expected = int(sys.argv[1])
print("python       :", sys.version.split()[0], sys.executable)
print("torch        :", torch.__version__)
print("torch_mlu    :", getattr(torch_mlu, "__version__", "unknown"))
print("NEUWARE_HOME :", os.environ.get("NEUWARE_HOME", "(unset)"))
print("CPU fallback :", "ENABLE_FALLBACK_TO_CPU=" + os.environ.get("ENABLE_FALLBACK_TO_CPU", "(unset)"))
available = torch.mlu.is_available()
count = torch.mlu.device_count() if available else 0
print("mlu avail    :", available)
print("device count :", count, f"(expected {expected})")
if not available:
    sys.exit("torch_mlu sees no MLU. Check the driver (cnmon), NEUWARE_HOME / LD_LIBRARY_PATH "
             "and MLU_VISIBLE_DEVICES.")
if count < expected:
    sys.exit(f"torch sees {count} card(s) but {expected} were requested; check MLU_VISIBLE_DEVICES.")
for index in range(count):
    try:
        name = torch.mlu.get_device_name(index)
    except Exception as exc:
        name = f"name unavailable ({type(exc).__name__})"
    try:
        free, total = torch.mlu.mem_get_info(index)
        memory = f"{free / 2**30:.1f} / {total / 2**30:.0f} GiB free"
    except Exception as exc:
        memory = f"memory query unavailable ({type(exc).__name__})"
    print(f"  mlu{index}: {name}, {memory}")
probe = torch.randn(256, 256, device="mlu")
print("fp32 matmul  : ok", float((probe @ probe).sum()))
# The training loop aggregates per-epoch statistics; both of these are guarded in the code,
# so a "no" here is a note, not a blocker.
try:
    torch.zeros(3, dtype=torch.float64, device="mlu").sum().item()
    print("fp64 on card : yes")
except Exception as exc:
    print("fp64 on card : NO --", type(exc).__name__, str(exc)[:120])
try:
    pinned = torch.zeros(16).pin_memory().is_pinned()
    print("pinned host  :", "yes (FLAGSCALE_PIN_MEMORY=true may be used)" if pinned else "no")
except Exception as exc:
    print("pinned host  : NO --", type(exc).__name__, str(exc)[:120])
is_cncl_available = getattr(torch.distributed, "is_cncl_available", None)
cncl = bool(is_cncl_available and is_cncl_available())
print("cncl         :", "yes" if cncl else "NO")
if expected > 1 and not cncl:
    sys.exit(f"{expected} cards were requested but the cncl backend is unavailable.")
try:
    from flagscale.platforms import get_platform

    platform = get_platform()
except Exception as exc:
    sys.exit(f"FlagScale did not register the mlu platform: {type(exc).__name__}: {exc}")
print("platform     :", platform.name(), "/ backend", platform.dist_backend())
if platform.name() != "mlu":
    sys.exit(f"FlagScale selected {platform.name()!r}, not 'mlu'.")
PY
} 2>&1 | tee "$LOGS/preflight.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "===== preflight failed (exit=$rc); not starting =====" >&2; exit "$rc"; fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then echo "===== preflight only; done ====="; exit 0; fi

cd "$ROOT" || exit 1

# ---- operator probe ----------------------------------------------------------------------
# Single process on the first listed card, no FlagScale or torchrun.  Extra arguments go to
# smoke.py here, not to Hydra.
if [ "${SMOKE:-0}" = 1 ]; then
  log="$LOGS/smoke-$config.log"
  echo "===== smoke $module/$config card=${cards%%,*} start $(date '+%F %T') ====="
  "$PY" "examples/$module/smoke.py" --config "$config" --device mlu ${extra[@]+"${extra[@]}"} 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  scan_log "$log" || { [ "$rc" -eq 0 ] && rc=3; }
  echo "===== smoke $module/$config exit=$rc end $(date '+%F %T') ====="
  exit "$rc"
fi

# ---- training ----------------------------------------------------------------------------
log="$LOGS/train-$config.log"
echo "===== $module/$config cards=$cards nproc=$nproc start $(date '+%F %T') ====="
"$PY" -m flagscale.run \
  --config-path "examples/$module/conf" --config-name train \
  action="${ACTION:-test}" "train=$config" \
  train.hardware.device_type=mlu train.hardware.distributed_backend=cncl \
  experiment.runner.nproc_per_node="$nproc" \
  'experiment.cmds.before_start="true"' \
  ${extra[@]+"${extra[@]}"} \
  2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
scan_log "$log" || { [ "$rc" -eq 0 ] && rc=3; }
echo "===== $module/$config exit=$rc end $(date '+%F %T') ====="
exit "$rc"
