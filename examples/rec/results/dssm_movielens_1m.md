# DSSM (two-tower) — MovieLens-1M, torch-rechub protocol

| Field | Value |
| --- | --- |
| Status | **finished** (10 epochs, the reference's budget) |
| Started / finished | 2026-11-04 14:04:22 → 14:08:13 (CST) |
| Wall clock | 3 m 51 s (training 224 s) |
| Config | `examples/rec/conf/train/dssm.yaml` |
| Reference document | [../EVIDENCE.md](../EVIDENCE.md) § DSSM |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, card 5, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 329,222 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/dssm-ml1m/rec/ckpts/dssm-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh rec dssm 5` |

## Protocol

The DSSM paper (Huang et al., CIKM 2013) trains on proprietary web-search click
logs, so the reproducible reference is torch-rechub's MovieLens-1M example
(`examples/matching/run_ml_dssm.py`), whose README reports **Hit@100 = 14.74 %**.

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `movielens_1m_dssm`: 6,040 users, 3,706 items, 3,952,516 train rows (1 positive + 3 popularity-sampled negatives per position), 6,040 test users (last item), full-catalogue retrieval | `[code]` |
| user tower | user_id, gender, age, occupation, zip + mean-pooled ≤50-item history (shares the movie table) | `[code]` |
| item tower | movie_id + first genre | `[code]` |
| embedding / towers | 16 dims, [256, 128, 64] Linear-BN-PReLU per tower, L2-normalised outputs | `[code]` |
| loss | point-wise BCE on `sigmoid(<u, v>)` | `[code]` |
| optimizer | Adam 1e-4, weight decay 1e-6 | `[code]` |
| batch size / epochs | 4096 / 10 | `[code]` |
| seed | 2022 | `[code]` |
| temperature | 1.0 (torch-rechub accepts 0.02 but its `forward` never applies it) | `[code]` |
| retrieval | exact cosine top-k over all 3,706 items (torch-rechub uses an approximate Annoy index) | `[ours]` |

## Curve (test)

| Epoch | Train loss | Hit@100 | NDCG@100 | Epoch time |
| --- | --- | --- | --- | --- |
| 1 | 0.6015 | 0.1172 | — | 24 s |
| 2 | 0.5668 | 0.1609 | — | 21 s |
| 3 | 0.5607 | 0.1810 | — | 21 s |
| 5 | 0.5561 | 0.1864 | — | 21 s |
| 7 | 0.5544 | 0.1897 | — | 21 s |
| 9 | 0.5534 | 0.1924 | — | 21 s |
| **10 (last)** | **0.5530** | **0.1921** | **0.0430** | 21 s |

## Results

| Metric | This run | torch-rechub README | Previous runs of the same config |
| --- | --- | --- | --- |
| Hit@100 = Recall@100 | **0.1921** | 0.1474 | 0.1921 (vendor, 2026-10-15), 0.1964 (Torch-FL boxing, 2026-10-22) |
| NDCG@100 | 0.0430 | — | 0.0430 / 0.0438 |
| Hit@10 / NDCG@10 | 0.0281 / 0.0121 | — | 0.0281 / 0.0121 |
| Hit@50 / NDCG@50 | 0.1127 / 0.0302 | — | 0.1127 / 0.0302 |

The run reproduces the 2026-10-15 vendor number to four decimals (0.1921), so
the current tree behaves exactly like the snapshot the earlier number came
from.  It is **+0.045 above the torch-rechub README**: the reference's README
number comes from a different negative-sampling draw and an approximate Annoy
index, and the EVIDENCE.md note already flags the reference as loosely
specified (the script prints top-10 while the README reports @100).

Throughput: 179,419 samples/s (21 s/epoch, 3.95 M rows), 3 m 51 s end to end.

## Notes

* `dssm` is no longer one of the six benchmark models — `two_tower` replaced it
  (see [README](../README.md)).  It is kept because the project's older metric
  table also tracks it, and it is the only model whose reference is a framework
  example rather than a paper.
* The earlier 0.1964 came from the Torch-FL boxing runtime; the 0.1921 here and
  on the 2026-10-15 vendor run agree, so the small difference is a runtime
  effect, not a code change.
* Exact retrieval instead of Annoy can only *raise* Hit@100 slightly; it does
  not explain 0.045.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | final weights |
| `checkpoint/history.json` | per-epoch train/test history |
| `checkpoint/metrics.json` | summary + full history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=dssm train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
