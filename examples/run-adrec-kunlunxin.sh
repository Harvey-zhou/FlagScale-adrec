#!/bin/bash
# Run one advertising (ads) or recommendation (rec) model on Kunlunxin cards with the
# vendor torch (XPytorch).
#
#   bash examples/run-adrec-kunlunxin.sh ads din 0
#   bash examples/run-adrec-kunlunxin.sh rec lightgcn_yelp2018 1
#   bash examples/run-adrec-kunlunxin.sh rec sasrec_beauty 0,1       # two cards, DDP
#   bash examples/run-adrec-kunlunxin.sh rec sasrec 0 train.epochs=1
#
# The vendor torch reports a Kunlunxin card as `cuda`, so the model, training and platform
# code are the ones used on an NVIDIA card; only the launch differs.  This is the same route
# as examples/retrieval/run-kunlunxin.sh.  Batch size, learning rate, epochs and the data
# split come from examples/<ads|rec>/conf/train/<config>.yaml, so every platform runs the
# same experiment.  Anything after the card list is passed on as a Hydra override.
#
# Environment:
#   VENDOR_ENV_SH    required: a script that activates the vendor torch environment
#   ADREC_DATA_ROOT  datasets root (default /flagos-ads/datasets or /flagos-rec/datasets)
#   ADREC_OUT_ROOT   checkpoints, logs and scratch (default $HOME/flagos-adrec); keep it on
#                    a disk with free space, a full root partition fails runs with ENOSPC
#   PY               python executable (default python)
#   DEVICE_TYPE      train.hardware.device_type (default cuda)
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
: "${VENDOR_ENV_SH:?set VENDOR_ENV_SH to the script that activates the Kunlunxin vendor torch environment}"
# shellcheck disable=SC1090
source "$VENDOR_ENV_SH"
PY=${PY:-python}

OUT=${ADREC_OUT_ROOT:-$HOME/flagos-adrec}/$module
LOGS=$OUT/logs
export TMPDIR="$OUT/tmp"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

export CUDA_VISIBLE_DEVICES="$cards"
export XPU_VISIBLE_DEVICES="$cards"
nproc=$(awk -F, '{print NF}' <<< "$cards")
export "$data_var"="${ADREC_DATA_ROOT:-${!data_var:-$data_default}}"
export "$ckpt_var"="$OUT/ckpts"
export FLAGSCALE_HOME="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Not Torch-FL: no operator is routed through FlagGems on this path.
export FLAGOS_USE_FLAGGEMS=false
export WANDB_MODE=${WANDB_MODE:-offline}

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
  train.hardware.device_type="${DEVICE_TYPE:-cuda}" train.hardware.distributed_backend=auto \
  experiment.runner.nproc_per_node="$nproc" \
  'experiment.cmds.before_start="true"' \
  ${extra[@]+"${extra[@]}"} \
  2>&1 | tee "$LOGS/train-$config.log"
rc=${PIPESTATUS[0]}
echo "===== $module/$config exit=$rc end $(date '+%F %T') ====="
exit "$rc"
