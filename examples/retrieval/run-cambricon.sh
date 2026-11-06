#!/bin/bash
# Run one retrieval model on Cambricon MLU cards (vendor torch_mlu + CNCL).
#
#   bash run-cambricon.sh bge_m3                      # card 0
#   bash run-cambricon.sh clip_vit_large 1            # card 1
#   bash run-cambricon.sh qwen3_reranker 0,1          # 2 cards, FSDP2 (the default for the reranker)
#   PREFLIGHT_ONLY=1 bash run-cambricon.sh qwen3_reranker 0,1   # probe the stack, do not train
#   SMOKE=1 bash run-cambricon.sh clip_vit_large 0    # one real batch: forward, backward, AdamW step
#   SMOKE=1 ALLOW_CPU_FALLBACK=1 bash run-cambricon.sh bge_m3 0 # list every operator that would run on the CPU
#
# Inside the Cambricon container (the host's Python is too old and has no NEUWARE):
#   bash examples/retrieval/cambricon-container.sh bash examples/retrieval/run-cambricon.sh bge_m3 0
#
# Only the model, the cards and the output location are passed.  Batch size, learning
# rate, epochs, max_length and the data split come from conf/train/<model>.yaml, so a
# Cambricon run is comparable with the A100 / PPU / Kunlunxin / Ascend ones.
#
# Stack: torch_mlu claims PyTorch's PrivateUse1 slot as `mlu` and
# flagscale/platforms/platform_mlu.py drives it (`device_type=mlu`, backend `cncl`), the
# same way run-ascend.sh drives torch_npu.  Torch-FL is not involved and
# torch_mlu.utils.gpu_migration is not used (it would report `cuda` for an MLU run).
#
# Parallel mode follows docs/retrieval/14-cambricon.md:
#   bge_m3, clip_vit_large   one card, fp32 master weights, bf16 through autocast
#   qwen3_reranker           FSDP2 on two cards, fp32 master weights, batch 32 per rank
# The reranker needs 8.19B x 16 bytes = 131 GB of weights, gradients and Adam state, which
# does not fit one 96 GB card.  If two cards run out of memory, the recorded fallback is
#   RERANKER_BATCH=16 bash run-cambricon.sh qwen3_reranker 0,1,2,3
# (64 rows per step either way).  That is a platform exception: write it into the table
# in docs/retrieval/14-cambricon.md before quoting the run.
#
# CPU fallback.  torch_mlu runs an operator it has no MLU kernel for on the CPU and only
# warns once per operator.  This script sets ENABLE_FALLBACK_TO_CPU=0 and also scans the
# log for torch_mlu's warning, so an unsupported operator fails the run instead of making
# a CPU run that looks like an MLU result.  ALLOW_CPU_FALLBACK=1 lifts both, for the
# operator probe only; such a run is not a result.
#
# Environment (all overridable):
#   FLAGSCALE_PYTHON_ENV  env holding torch + torch_mlu (default /torch/venv3/pytorch, the
#                         26.08 container); its bin/ goes first on PATH
#   RETRIEVAL_EXTRA_PYPATH  directory of extra pip packages (transformers, tokenizers,
#                         safetensors, Pillow, pyarrow, webdataset) installed with
#                         `pip install --target`, so the vendor torch is never replaced
#   NEUWARE_HOME          Cambricon SDK root (default /usr/local/neuware when it exists)
#   WORK_ROOT             default /srv/$USER; `/` has about 100 GB free, /srv about 6 TB
#   FLAGSCALE_OUTPUT_DIR  checkpoints and logs (default $WORK_ROOT/flagos-retrieval-cambricon/<model>)
#   PYTORCH_MLU_ALLOC_CONF  default expandable_segments:True (P800 failed the reranker for lack of it)
#   WANDB_MODE            default offline
#   ACTION                FlagScale action (default test, which runs in the foreground)
#   RERANKER_BATCH        per-rank batch for the reranker; set only for the 4-card fallback
set -uo pipefail
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT" || exit 1

model=${1:?usage: run-cambricon.sh <bge_m3|clip_vit_large|qwen3_reranker> [cards, e.g. 0 or 0,1]}
case "$model" in
  bge_m3|clip_vit_large) default_cards=0 ;;
  qwen3_reranker) default_cards=0,1 ;;
  *) echo "model must be bge_m3, clip_vit_large or qwen3_reranker, got '$model'" >&2
     echo "(qwen3_embedding and qwen3_vl_embedding are outside the Cambricon scope)" >&2
     exit 2 ;;
esac
cards=${2:-$default_cards}
extra=("${@:3}")
nproc=$(awk -F, '{print NF}' <<< "$cards")
if [ ! -f "examples/retrieval/conf/train/$model.yaml" ]; then
  echo "no such config: examples/retrieval/conf/train/$model.yaml" >&2
  exit 2
fi

# ---- interpreter -------------------------------------------------------------------------
VN=${FLAGSCALE_PYTHON_ENV:-/torch/venv3/pytorch}
PY="$VN/bin/python"
if [ ! -x "$PY" ]; then
  echo "no python at $PY; set FLAGSCALE_PYTHON_ENV to the env holding torch_mlu" >&2
  echo "(inside the container it is /torch/venv3/pytorch; outside it, use cambricon-container.sh)" >&2
  exit 2
fi
# FlagScale's `before_start` sources ${FLAGSCALE_PYTHON_ENV}/bin/activate and then runs bare
# `torchrun`.  The launch below replaces before_start, so PATH has to carry the env itself.
export FLAGSCALE_PYTHON_ENV="$VN"
export PATH="$VN/bin:$PATH"
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "$PY is older than Python 3.10; run inside the Cambricon container" >&2
  exit 2
fi

# ---- vendor stack ------------------------------------------------------------------------
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
# The CUDA allocator's setting is `PYTORCH_ALLOC_CONF`; torch_mlu reads its own variable.
# Whether 1.34.1 honours it is checked by the preflight (the probe prints the value, and
# a run that fragments will say so in its memory log).
export PYTORCH_MLU_ALLOC_CONF="${PYTORCH_MLU_ALLOC_CONF:-expandable_segments:True}"
# Pin the platform here, so a FLAGSCALE_DEVICE / FS_PLATFORM left in the shell cannot
# override what conf/train.yaml derives from device_type.
export FLAGSCALE_DEVICE=mlu FS_PLATFORM=mlu FLAGSCALE_DISTRIBUTED_BACKEND=cncl

WORK_ROOT=${WORK_ROOT:-/srv/${USER:-user}}
OUT=${FLAGSCALE_OUTPUT_DIR:-$WORK_ROOT/flagos-retrieval-cambricon/$model}
LOGS=$OUT/logs
export TMPDIR="$OUT/tmp"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" || { echo "cannot create $OUT" >&2; exit 2; }

export FLAGSCALE_HOME="$ROOT"
export PYTHONPATH="$ROOT${RETRIEVAL_EXTRA_PYPATH:+:$RETRIEVAL_EXTRA_PYPATH}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
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
echo "===== preflight: $PY model=$model cards=$cards nproc=$nproc ====="
{
  # The card clock is part of every throughput figure: the cluster caps it at 1200 MHz
  # (1600 default), so keep the cnmon output with the log.
  if command -v cnmon >/dev/null 2>&1; then cnmon || true; fi
  "$PY" - "$model" <<'PY'
import importlib.util
import sys

missing = [m for m in ("torch", "torch_mlu", "omegaconf", "hydra", "transformers", "tokenizers",
                       "safetensors", "PIL", "pyarrow", "webdataset", "numpy")
           if importlib.util.find_spec(m) is None]
print("python:", sys.version.split()[0], sys.executable)
if missing:
    sys.exit(f"missing packages in this environment: {missing}. Install the non-vendor ones with "
             "`pip install --target $RETRIEVAL_EXTRA_PYPATH ...`; never replace torch or torch_mlu.")
PY
} 2>&1 | tee "$LOGS/preflight-$model.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "===== preflight failed (exit=$rc); not starting =====" >&2; exit "$rc"; fi

# Paths named by the training config must exist, otherwise the run fails after minutes
# of model loading.  Read through OmegaConf, which resolves the ${...} references.
"$PY" - "examples/retrieval/conf/train/$model.yaml" <<'PY' 2>&1 | tee -a "$LOGS/preflight-$model.log"
import os
import sys

from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
paths = {"model.model_path": cfg.model.model_path, "data.path": cfg.data.path}
for key in ("validation_path", "test_path"):
    if cfg.data.get(key):
        paths[f"data.{key}"] = cfg.data[key]
bad = {k: v for k, v in paths.items() if not os.path.exists(str(v))}
for k, v in paths.items():
    print(f"  {k:22} {'ok     ' if k not in bad else 'MISSING'} {v}")
if bad:
    sys.exit("paths missing; mount them (cambricon-container.sh: RETRIEVAL_MODELS_HOST / "
             "RETRIEVAL_DATA_HOST) or link them at the paths above")
PY
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then echo "===== path check failed (exit=$rc); not starting =====" >&2; exit "$rc"; fi

# The stack probe: autocast bf16, fp64 all_reduce, FSDP2 operators, and with more than one
# card a small FSDP2 model.  The retrieval code's silent-failure risks are checked here.
echo "===== stack probe (cards=$cards) ====="
if [ "$nproc" -gt 1 ]; then
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
    examples/retrieval/probe_mlu.py 2>&1 | tee "$LOGS/probe-$model.log"
else
  "$PY" examples/retrieval/probe_mlu.py 2>&1 | tee "$LOGS/probe-$model.log"
fi
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] && [ "${ALLOW_CPU_FALLBACK:-0}" != 1 ]; then
  echo "===== stack probe failed (exit=$rc); not starting =====" >&2
  exit "$rc"
fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then echo "===== preflight only; done ====="; exit "$rc"; fi

# ---- operator probe on the real model ----------------------------------------------------
# One card, no FlagScale or torchrun: load the model, build one batch, forward, backward and
# one AdamW step, so an operator gap is named before a long run.
if [ "${SMOKE:-0}" = 1 ]; then
  log="$LOGS/smoke-$model.log"
  echo "===== smoke $model card=${cards%%,*} start $(date '+%F %T') ====="
  "$PY" examples/retrieval/smoke_model.py "$model" --device mlu ${extra[@]+"${extra[@]}"} 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  scan_log "$log" || { [ "$rc" -eq 0 ] && rc=3; }
  echo "===== smoke $model exit=$rc end $(date '+%F %T') ====="
  exit "$rc"
fi

# ---- training ----------------------------------------------------------------------------
# bge/clip run on one card (the baselines do); only the reranker shards.
PARALLEL=ddp
EXTRA_ARGS=()
if [ "$model" = qwen3_reranker ]; then
  PARALLEL=fsdp2
  if [ "$nproc" -lt 2 ]; then
    echo "qwen3_reranker needs at least 2 cards (8.19B x 16 bytes does not fit one 96 GB card)" >&2
    exit 2
  fi
  if [ -n "${RERANKER_BATCH:-}" ]; then
    # Rows per step stay 64 only for 4 cards x 16; say so, so the table gets the exception.
    echo "NOTE: RERANKER_BATCH=${RERANKER_BATCH}, $nproc cards => $((RERANKER_BATCH * nproc)) rows per step." \
         "This is a platform exception to record in docs/retrieval/14-cambricon.md."
    EXTRA_ARGS+=("train.batch_size=${RERANKER_BATCH}" "train.gradient_accumulation_steps=1")
  fi
fi

log="$LOGS/train-$model.log"
echo "===== $model cards=$cards parallel=$PARALLEL nproc=$nproc start $(date '+%F %T') ====="
"$PY" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action="${ACTION:-test}" "train=$model" \
  train.parallel=$PARALLEL train.master_dtype=fp32 \
  experiment.runner.nproc_per_node="$nproc" \
  train.hardware.device_type=mlu train.hardware.distributed_backend=cncl \
  train.hardware.autocast_device=mlu train.hardware.use_flaggems=false \
  'experiment.cmds.before_start="true"' \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  ${extra[@]+"${extra[@]}"} \
  train.output_dir="$OUT" experiment.exp_dir="$LOGS" \
  hydra.run.dir="$LOGS/hydra" \
  2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
scan_log "$log" || { [ "$rc" -eq 0 ] && rc=3; }
echo "===== $model exit=$rc end $(date '+%F %T') ====="
exit "$rc"
