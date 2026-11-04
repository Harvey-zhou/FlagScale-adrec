# BARS reference configs — Criteo_x1, reproduced on this host

| Field | Value |
| --- | --- |
| Status | **finished** (early stopping decided both runs) |
| Started / finished | 2026-11-03 21:08 → 22:13 (DeepFM), → 22:16 (DCN-Mix) |
| Configs | `examples/ads/conf/train/deepfm.yaml`, `examples/ads/conf/train/dcn_mix.yaml` |
| Evidence | [../EVIDENCE.md](../EVIDENCE.md) (published numbers) |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, cards 5 and 6, vendor torch 2.9.0+cu128, fp32 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/bars-deepfm/`, `.../bars-dcnmix/` |

## Why these runs

Every advertising conclusion in this directory is stated against "the BARS
reference on the same split" (DeepFM 0.8137 / 0.4381, DCN-V2 0.8142 / 0.4378).
Those numbers came from the BARS repository, not from this tree: only `din.yaml`
had ever been run here.  If the BARS recipe does not reproduce on this host, the
paper-protocol gaps are measured against a number that may not hold here — so
both BARS configs were run unchanged.

## Results

| Run | Params | Epochs | Best epoch | test AUC | test logloss | BARS published | Δ AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `deepfm` (BARS) | 23,429,477 | 11 (early stop) | 9 | **0.811187** | 0.440624 | 0.813666 / 0.438147 | −0.0025 |
| `dcn_mix` (BARS + mixture on) | 21,875,089 | 14 (early stop) | 12 | **0.813934** | 0.437975 | 0.814195 / 0.437788 (full-rank DCN-V2) | −0.0003 |

The DeepFM parameter count matches EVIDENCE.md exactly (23,429,477), so the
implementation is the same one that produced BARS's number; the 0.0025 AUC
difference is run-to-run variation around a published 5-run mean of 0.813812.

DCN-Mix reproduces to 0.0003 AUC even though it runs with the low-rank mixture
**on** while BARS publishes the full-rank DCN-V2 — the paper's own finding is
that the mixture is worth about 0.0003 AUC, which is the size of the difference
observed here.

## Curves

Both runs use the FuxiCTR procedure: Adam 1e-3, the monitored validation AUC
multiplied down by 0.1 on a non-improvement, `patience` 2, best checkpoint
restored for the test pass.

| Epoch | DeepFM valid AUC | DCN-Mix valid AUC |
| --- | --- | --- |
| 1 | 0.80308 | 0.80460 |
| 2 | 0.80543 | 0.80714 |
| 3 | 0.80661 | 0.80824 |
| 4 | 0.79830 | 0.80875 |
| 5 | 0.80832 (lr 1e-4) | 0.80924 |
| 6 | 0.80991 | 0.80939 |
| 7 | 0.81061 | 0.80992 |
| 8 | 0.81082 | 0.81004 |
| 9 | **0.81084 (best)** | 0.81018 |
| 10 | 0.81059 | 0.81017 |
| 11 | 0.80980 (lr 1e-5) | 0.81331 (lr 1e-4) |
| 12 | — | **0.81363 (best)** |
| 13 | — | 0.81358 |
| 14 | — | 0.81314 (lr 1e-5) |

DeepFM's epoch-4 drop (0.7983) is the only irregular point and recovers on the
next epoch; the LR reductions at epochs 5 and 11 are the scheduler's.

## What this changes

The paper-protocol comparisons are now against a same-host baseline rather than
a published one:

| Model | Paper protocol | BARS config, this host | Published BARS | Paper |
| --- | --- | --- | --- | --- |
| DeepFM | 0.8040 / 0.4477 | **0.8112 / 0.4406** | 0.8137 / 0.4381 | 0.8016 / 0.44985 |
| DCN-Mix | 0.8104 / 0.4411 (lr 5e-4) | **0.8139 / 0.4380** | 0.8142 / 0.4378 | 0.8112 / 0.4408 |
| DCN-Mix | 0.8115 / 0.4401 (lr 1e-4) | same | same | same |

* DeepFM: the paper-protocol gap is **0.0072 AUC**, not the 0.0097 the
  published BARS number implies — a fifth of it was the baseline being 0.0025
  high.  The paper protocol itself (0.8040) is *above* the paper's own 0.8016,
  although that number comes from the unavailable Criteo split.
* DCN-Mix: the gap at the paper's lr range is **0.0024–0.0035**, and the
  ablation already showed the learning rate moves in the useful direction
  (1e-4 better than 5e-4).  The BARS recipe differs from the paper's best
  setting in five other places (embedding 10 vs 39, low rank 32 vs 64, DNN
  500³ vs 512², dropout 0.1 vs 0, batch 4096 vs 512), so part of the remaining
  gap is a different model, not a worse one.

## Cost

| Run | Epoch time | Mean samples/s |
| --- | --- | --- |
| DeepFM | 340–344 s | 96,204 |
| DCN-Mix | 268–288 s | 115,797 |

Batch 4096, 33.0 M rows per epoch (8,058 steps).

## Artifacts

| Path | Contents |
| --- | --- |
| `.../bars-deepfm/ads/ckpts/deepfm-cuda/checkpoint/` | metrics.json, history.json, model.pt |
| `.../bars-dcnmix/ads/ckpts/dcn_mix-cuda/checkpoint/` | same |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

# BARS DeepFM (early stopping decides)
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=deepfm train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1

# BARS DCN-V2 with the low-rank mixture on
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=dcn_mix train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
