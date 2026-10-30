# DeepFM — Criteo_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 6; best epoch 1) |
| Started / finished | 2026-10-30 10:04:30 → 10:38 (CST) |
| Wall clock | 33 min (6 epochs × ~300 s + ~7 min of data loading) |
| Config | `examples/ads/conf/train/deepfm_paper.yaml` |
| Code commit | launched from the working tree at 10:04, committed 10:13 as `1ed6f6e`; training code and config are identical to that commit (only the results doc/README came later) |
| Device | Kunlunxin P800, card 2, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 23,427,077 |
| Output | `/data2/jiangxuhui/logs/ads-paper-deepfm/ckpts/deepfm_paper-cuda/checkpoint/` |
| Command | `launch-ads-paper.sh 2 deepfm_paper /data2/jiangxuhui/logs/ads-paper-deepfm` |

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| dataset | `criteo_x1` (reczoo): 13 numeric + 26 categorical fields; train 33,003,326 / valid 8,250,124 / test 4,587,167 rows (7:2:1, AFN split) | — |
| model | FM (order-1 + order-2 inner products) + DNN over the same field embeddings | `[paper]` §2.1 |
| embedding / FM latent dim | 10 | `[paper]` §3.1.4 |
| DNN | 400-400-400, ReLU | `[paper]` §3.1.4 |
| dropout | 0.5 | `[paper]` §3.1.4 |
| BatchNorm | off | `[ours]` the paper describes no BatchNorm (BARS adds it) |
| optimizer | Adam | `[paper]` §3.1.4 |
| learning rate | 1e-3, constant | `[ours]` the paper states no lr |
| weight decay / embedding L2 | 0 / 0 | `[ours]` the paper uses dropout only |
| gradient clipping | none | `[ours]` |
| batch size | 4096 | `[ours]` not in the paper; Criteo convention |
| epochs | 100 (cap) | `[ours]` |
| early stopping | stop after 5 epochs without validation AUC improvement | `[ours]` the paper gives no rule |
| metric | AUC / logloss | `[paper]` §3.1.2 |
| seed | 2021 | `[ours]` |

## Data-split caveat

The paper uses a random 90/10 split of the 45M-row Kaggle Criteo file, without
publishing the split or a seed.  `criteo_x1` is the reczoo release with a 7:2:1
split (following AFN, AAAI 2020), so **the paper's 0.8007 / 0.45083 is not a
target that can be reproduced here**.  The comparable reference on this split is
the BARS/FuxiCTR `deepfm.yaml` run: AUC 0.8137 / logloss 0.4381 (dropout 0.1,
BatchNorm, embedding L2 1e-5).

## Curve

Each epoch is ~300 s (33M rows / 4096 = 8,058 steps at ~110k samples/s).

| Epoch | Train loss | Valid AUC | Valid logloss |
| --- | --- | --- | --- |
| **1 (best)** | **0.4543** | **0.8036** | **0.4482** |
| 2 | 0.4373 | 0.8016 | 0.4509 |
| 3 | 0.4242 | 0.7978 | 0.4595 |
| 4 | 0.4157 | 0.7930 | 0.4648 |
| 5 | 0.4080 | 0.7894 | 0.4741 |
| 6 | 0.4017 | 0.7856 | 0.4819 |

Early stopping fired at epoch 6: the best validation AUC was epoch 1 and nothing
improved for the next 5 epochs.  Training loss keeps falling while validation
AUC falls every epoch — the paper's configuration overfits from the first epoch
on this 33M-row training set.

## Results

| Metric | This run (best epoch 1) | BARS `deepfm.yaml` (11 epochs) | Paper (different split) |
| --- | --- | --- | --- |
| test AUC | 0.8040 | 0.8137 | 0.8007 |
| test logloss | 0.4477 | 0.4381 | 0.45083 |
| valid AUC | 0.8036 | — | — |

Against the only comparable target on this data (BARS) the paper-settings run is
0.0097 AUC lower and 0.0096 logloss higher, while it is close to the paper's own
number — but that number comes from a 90/10 split that cannot be reproduced, so
it is context, not a target.

**Why it underperforms BARS is not established.**  The paper-silent differences
are BatchNorm, embedding L2 (both added by FuxiCTR) and the learning rate /
early-stop rule (ours).  Any of them could matter; no ablation has been run.  A
cheap next check would be a 2–3 epoch sweep of the learning rate (1e-4, 3e-4),
since the model already peaks after one epoch at 1e-3.

## Notes

* BatchNorm is the main model-level difference from `deepfm.yaml`: the paper's
  §3.1.4 description is dropout + 400-400-400 + ReLU only, while FuxiCTR adds
  BatchNorm and embedding L2.  Both differences are recorded rather than hidden.
* Early stopping and the learning rate are not paper-specified; 5-epoch patience
  on validation AUC and Adam 1e-3 are our choices.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation-AUC epoch weights (93 MB, rewritten on improvement) |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | written at the end: summary + full history |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=deepfm_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
