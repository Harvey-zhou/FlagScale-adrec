#!/bin/bash
# Run one advertising (ads) or recommendation (rec) model on Kunlunxin P800 cards.
#
#   bash examples/run-adrec-kunlunxin.sh ads din 0
#   bash examples/run-adrec-kunlunxin.sh rec lightgcn_yelp2018 1
#   bash examples/run-adrec-kunlunxin.sh rec sasrec_beauty 0,1       # two cards, DDP
#   bash examples/run-adrec-kunlunxin.sh rec sasrec 0 train.epochs=1 # extra Hydra overrides
#   PREFLIGHT_ONLY=1 bash examples/run-adrec-kunlunxin.sh ads din 0  # check the stack, do not train
#
# The vendor stack (xpytorch / XMLIR) rewrites torch's CUDA symbols onto the Kunlunxin
# runtime at import time, so a card is `device='cuda'` and the model, training and platform
# code are the ones used on an NVIDIA card.  Torch-FL is NOT involved on this path.
#
# Three things have to be right before torch sees a card, and getting any of them wrong
# looks like a missing driver rather than a missing variable:
#
#   * the interpreter must be the venv that carries torch_xmlir.  A plain conda env with
#     upstream torch fails with "The NVIDIA driver on your system is too old";
#   * LD_LIBRARY_PATH must contain <venv>/xcudart/lib, the CUDA-ABI shim over the XPU driver;
#   * DISABLE_XPYTORCH must be unset.  It is injected with value 1 on some hosts, and with it
#     set the vendor stack stays off.
#
# This script does all three and then asserts that torch really sees the expected number of
# cards, so a misconfigured environment fails here instead of midway through training.
# Batch size, learning rate, epochs and the data split come from
# examples/<ads|rec>/conf/train/<config>.yaml, so every platform runs the same experiment.
#
# Environment:
#   ADREC_VENV       the venv holding torch_xmlir; tried in order: this variable,
#                    <repo>/venv, /data2/jiangxuhui/flagscale-adrec-20261014/venv
#   VENDOR_ENV_SH    optional: sourced first, for a host that activates the stack its own way
#   ADREC_DATA_ROOT  datasets root (default /flagos-ads/datasets or /flagos-rec/datasets)
#   ADREC_OUT_ROOT   checkpoints, logs and scratch (default $HOME/flagos-adrec); keep it on a
#                    disk with free space, a full root partition fails runs with ENOSPC
#   OMP_NUM_THREADS  default 1.  The container's cgroup may grant only a few CPU cores, and
#                    the per-model configs also ask for dataloader workers.
#   ACTION           FlagScale action (default test, which runs in the foreground)
set -uo pipefail

module=${1:?usage: run-adrec-kunlunxin.sh <ads|rec> <config> [cards] [hydra overrides...]}
config=${2:?config name, e.g. din, lightgcn_yelp2018, sasrec_beauty}
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

if [ -n "${VENDOR_ENV_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "$VENDOR_ENV_SH"
fi

# ---- interpreter -------------------------------------------------------------------------
VENV=""
for candidate in "${ADREC_VENV:-}" "$ROOT/venv" /data2/jiangxuhui/flagscale-adrec-20261014/venv; do
  if [ -n "$candidate" ] && [ -x "$candidate/bin/python" ]; then VENV=$candidate; break; fi
done
if [ -z "$VENV" ]; then
  echo "no venv with a python found; set ADREC_VENV to the venv that has torch_xmlir" >&2
  echo "tried: \${ADREC_VENV}, $ROOT/venv, /data2/jiangxuhui/flagscale-adrec-20261014/venv" >&2
  exit 2
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"

# ---- vendor stack ------------------------------------------------------------------------
# DISABLE_XPYTORCH=1 keeps the stack off; it is injected on some hosts.  CUDA_VISIBLE_DEVICES
# is cleared because XPU_VISIBLE_DEVICES is the reliable selector and the two should not
# disagree.
unset DISABLE_XPYTORCH CUDA_VISIBLE_DEVICES
export XPU_VISIBLE_DEVICES="$cards"
export LD_LIBRARY_PATH="$VENV/xcudart/lib:/usr/local/xpu/so:/usr/local/xpu/lib:${LD_LIBRARY_PATH:-}"

OUT=${ADREC_OUT_ROOT:-$HOME/flagos-adrec}/$module
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

# ---- preflight ---------------------------------------------------------------------------
# Asserts the stack is live and reports the two dtype/host-memory capabilities the training
# code depends on, so a gap is named here rather than inferred from a later failure.
echo "===== preflight: $PY (cards=$cards) ====="
"$PY" - "$nproc" <<'PY'
import sys

import torch

expected = int(sys.argv[1])
print("torch        :", torch.__version__)
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
print("cuda avail   :", available)
print("device count :", count, f"(expected {expected})")
if not available:
    sys.exit("vendor stack is not active: torch sees no card. Check that this python has "
             "torch_xmlir, that LD_LIBRARY_PATH carries <venv>/xcudart/lib, and that "
             "DISABLE_XPYTORCH is unset.")
if count < expected:
    sys.exit(f"torch sees {count} card(s) but {expected} were requested; check XPU_VISIBLE_DEVICES.")
for index in range(count):
    free, total = torch.cuda.mem_get_info(index)
    print(f"  xpu{index}: {free / 2**30:6.1f} / {total / 2**30:.0f} GiB free")
probe = torch.randn(256, 256, device="cuda")
print("fp32 matmul  : ok", float((probe @ probe).sum()))
# The training loop aggregates per-epoch statistics; both of these are guarded in the code,
# so a "no" here is a note, not a blocker.
try:
    torch.zeros(3, dtype=torch.float64, device="cuda").sum().item()
    print("fp64 on card : yes")
except Exception as exc:
    print("fp64 on card : NO --", type(exc).__name__, str(exc)[:120])
try:
    print("pinned host  :", "yes" if torch.zeros(16).pin_memory().is_pinned() else "no")
except Exception as exc:
    print("pinned host  : NO --", type(exc).__name__, str(exc)[:120])
PY
rc=$?
if [ "$rc" -ne 0 ]; then echo "===== preflight failed (exit=$rc); not starting training =====" >&2; exit "$rc"; fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then echo "===== preflight only; done ====="; exit 0; fi

# Without sparse tensor kernels LightGCN propagates with index_select + index_add, which is
# mathematically identical.
case "$config" in
  lightgcn*) if [[ "${extra[*]:-}" != *train.model.propagation=* ]]; then extra+=("train.model.propagation=index_add"); fi ;;
esac

echo "===== $module/$config cards=$cards nproc=$nproc start $(date '+%F %T') ====="
cd "$ROOT" || exit 1
"$PY" -m flagscale.run \
  --config-path "examples/$module/conf" --config-name train \
  action="${ACTION:-test}" "train=$config" \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  experiment.runner.nproc_per_node="$nproc" \
  'experiment.cmds.before_start="true"' \
  ${extra[@]+"${extra[@]}"} \
  2>&1 | tee "$LOGS/train-$config.log"
rc=${PIPESTATUS[0]}
echo "===== $module/$config exit=$rc end $(date '+%F %T') ====="
exit "$rc"
