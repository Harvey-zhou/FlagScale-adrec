# DIN — MovieLens-20M

| Field | Value |
| --- | --- |
| Status | **finished** (9 epochs, best epoch 9); BF16 counterpart running |
| Started / finished | 2026-11-04 14:21 → 2026-11-05 02:40 (CST) |
| Wall clock | 12 h 19 m |
| Config | `examples/ads/conf/train/din_ml20m.yaml` |
| Dataset | `examples/ads/prepare_movielens20m_din.py` → `/data2/jiangxuhui/datasets/movielens_20m_din` |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, card 5, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 3,033,090 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/din-ml20m/ads/ckpts/din_ml20m-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh ads din_ml20m 5` |

## Protocol — a project benchmark, not a paper reproduction

The DIN paper (KDD'18) uses Amazon and Alibaba; **it has no MovieLens
experiment**, so there is no paper number for this pair.  The dataset is the
authors' Amazon protocol applied to MovieLens-20M, and the model/optimiser is
the BARS/FuxiCTR `din.yaml` recipe:

| Setting | Value | Source |
| --- | --- | --- |
| interactions | 20,000,263 ratings, 138,493 users, 27,278 movies | `[code]` |
| split | leave-last-out: train = positions 1..n-2, test = the last item; one uniformly sampled negative per positive | `[ours]` the authors' Amazon protocol |
| rows | 39,446,554 train (1:1 pos/neg), 276,986 test | — |
| history | the **most recent** 100 items (Amazon's histories average 13, ML-20M's ~145, so the first 100 would drop recent behaviour) | `[ours]`, deviation recorded |
| item / category vocab | 26,746 / 22 (category = first genre) | — |
| model | embedding 64, attention [512, 256] Dice + dropout 0.1, softmax attention, DNN [1024, 512, 256] Dice + BN + dropout 0.1 | `[code]` BARS |
| optimiser | Adam 5e-4, embedding L2 5e-3, grad clip 10 | `[code]` BARS |
| batch / epochs | **1024** / 9, no early stopping | `[ours]` — the paper's batch 32 would be ~2 h per epoch and ~100 h for 50 |
| model selection | test split (there is no validation split) | `[ours]`, as for Amazon |

## Curve

| Epoch | Train loss | gAUC | AUC | logloss | Epoch time |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.2385 | 0.9725 | 0.9769 | 0.1955 | 1 h 22 m |
| 2 | 0.2286 | 0.9731 | 0.9777 | 0.19395 | 1 h 22 m |
| 3 | 0.2277 | 0.9739 | 0.9778 | 0.19227 | 1 h 21 m |
| 4 | 0.2277 | 0.9735 | 0.9781 | 0.19096 | 1 h 21 m |
| 5 | — | 0.9812 | 0.9850 | — | 1 h 21 m |
| 6 | — | 0.9819 | 0.9851 | — | 1 h 21 m |
| 7 | — | 0.9819 | 0.9850 | — | 1 h 21 m |
| 8 | — | 0.9839 | 0.9868 | — | 1 h 21 m |
| **9 (best)** | — | **0.9839** | **0.9869** | 0.1436 | 1 h 21 m |

The curve is **still improving at the last epoch**, so the 9-epoch budget
truncates the run rather than overfitting it (unlike Amazon, where the same
recipe peaks at epoch 7 and then declines).  A longer budget would very likely
raise the number further.

## Results

| Metric | FP32 (best epoch 9) | BF16 | Paper |
| --- | --- | --- | --- |
| gAUC | **0.9839** | running | none — no MovieLens experiment in the paper |
| AUC | 0.9869 | running | — |
| logloss | 0.1436 | running | — |

gAUC 0.98 is high because the task is "the true next movie against one random
movie out of 26.7 k": a well-trained model separates those easily.  The number
is **not comparable** with the Amazon Electronics result (0.8645), where the
candidate set is one negative sampled per user position from a 63 k-item
catalogue with much sparser histories.

Throughput: 8,056 samples/s (1 h 21 m per epoch over 39.4 M rows), 12 h 19 m
total.

## Notes

* Memory footprint: the dataset is 31 GB on disk and ~63 GB in host memory as
  int64 tensors; the two loader workers share the pages (copy-on-write), so the
  process RSS reads ~67 GB rather than 3× that.
* `max_len: 100` is the BARS value; the ML-20M history is truncated to the most
  recent 100 items.  Sampling one negative per positive keeps the 1:1 balance
  of the Amazon protocol.
* The model has no user embedding (the released DIN declares one but never uses
  it), so the user id is only carried as the gAUC grouping key.

## Artifacts

| Path | Contents |
| --- | --- |
| `checkpoint/model.pt` | best gAUC epoch weights (23 MB) |
| `checkpoint/history.json` | per-epoch history |
| `checkpoint/metrics.json` | summary + full history |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python examples/ads/prepare_movielens20m_din.py
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=din_ml20m train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
