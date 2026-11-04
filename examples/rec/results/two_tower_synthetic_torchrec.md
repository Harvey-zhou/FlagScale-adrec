# Two-Tower (TorchRec) — synthetic MovieLens-20M-format data

| Field | Value |
| --- | --- |
| Status | **finished** (100 iterations, the example's full budget) |
| Started / finished | 2026-11-03 19:29:38 → 19:29:50 (CST) |
| Wall clock | 12 s (training 4.6 s) |
| Config | `examples/rec/conf/train/two_tower.yaml` |
| Reference document | [../EVIDENCE.md](../EVIDENCE.md) § Two-tower |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, card 3, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 134,250,880 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/two-tower/rec/ckpts/two_tower-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh rec two_tower 3` |

## Settings

The config is a native port of TorchRec's `examples/retrieval`
(`modules/two_tower.py`, `two_tower_train.py`), so every value comes from that
example rather than from a paper.

| Setting | Value | Source |
| --- | --- | --- |
| data | `RandomRecDataset` in the MovieLens-20M format, drawn from `seed` on the host; no file is read | `[code]` |
| embedding tables | 2 × 1,048,576 × 64 (`EmbeddingBagCollection`), dense gradients | `[code]`; `[ours]` dense rather than FBGEMM |
| towers | ReLU MLP [128, 64] per tower, inner-product logits | `[code]` |
| loss | BCE-with-logits | `[code]` |
| optimizer | Adam 1e-2 on the MLPs, row-wise Adagrad 1e-2 on the tables | `[code]` |
| batch size | 32 | `[code]` |
| iterations | 100 (= 1 epoch here) | `[code]` |
| cached batches | 10, cycled (the example's `num_generated_batches`) | `[code]` |
| seed | 42 | `[ours]`, matches the CPU reference in EVIDENCE.md |

## Curve (running training loss, logged every 10 iterations)

| Step | Running loss |
| --- | --- |
| 10 | 0.6984 |
| 20 | 0.6953 |
| 30 | 0.6822 |
| 40 | 0.6368 |
| 50 | 0.6012 |
| 60 | 0.5753 |
| 70 | 0.5560 |
| 80 | 0.5416 |
| 90 | 0.5304 |
| **100** | **0.5215** |

## Results

The data are random, so there is no accuracy target.  EVIDENCE.md fixes the
acceptance as (1) the loss trajectory against a CUDA run of the same config and
seed and (2) the internal check that the cycled training batches fall below
ln 2 = 0.693 while the held-out batches stay near it.

| Metric | This run (P800) | CPU reference, seed 42 | Check |
| --- | --- | --- | --- |
| running loss after 100 steps | 0.5215 | 0.521 | match |
| training-set loss (10 cached batches) | 0.4402 | 0.440 | match |
| held-out loss (10 fresh batches) | 0.6923 | 0.6923 | ≈ ln 2, as expected for random labels |
| parameters | 134,250,880 | 134,250,880 | exact |

Both conditions hold: the optimiser memorises the cycled batches (0.693 → 0.440)
while generalisation on freshly drawn random labels remains at chance, which is
exactly the behaviour of the TorchRec example on this data.  The trajectory is
also equal to the CPU reference at every logged point, so the port is
numerically equivalent on this stack.

Throughput: 983.6 samples/s (3.25 s for 100 steps at batch 32).  The run is
dominated by the 512 MiB of embedding tables and the dense table gradients, not
by the MLPs.

## Notes

* **First attempt never started.**  It was launched on card 1, where the vendor
  JIT tried to write its scratch directory `/tmp/xpujitc-temp` — owned by another
  user with mode 700 on this host — and the first matmul hung.  `JITC_TEMP_DIR`
  (and a per-job `XBLAS_JITC_CACHE_PATH`) fixes it; card 1 itself was left wedged
  by the killed process and every later allocation on it hung, so the run used
  card 3.  This is a host sharing issue, not a model issue.
* The example's own trainer is not used: this is a native port (no
  `DistributedModelParallel`, no FBGEMM, no pipelined trainer, `nn.Embedding`
  times the 0/1 bag length instead of `EmbeddingBag` — identical for one id per
  feature).  See EVIDENCE.md for the full deviation list.
* The FAISS index and TorchRec's retrieval module are not ported, so there is no
  Hit@k to report; the model is accepted on the loss behaviour above.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | final weights (512 MiB) |
| `checkpoint/history.json` | the single epoch's train/test record |
| `checkpoint/metrics.json` | summary + full history |
| `checkpoint/config.json` | model config |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=two_tower train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
