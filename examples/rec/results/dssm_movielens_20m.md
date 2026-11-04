# DSSM (two-tower) — MovieLens-20M

| Field | Value |
| --- | --- |
| Status | **finished** (10 epochs) |
| Started / finished | 2026-11-04 14:24:29 → 15:37:05 (CST) |
| Wall clock | 1 h 12 m |
| Config | `examples/rec/conf/train/dssm_ml20m.yaml` |
| Dataset | `examples/rec/prepare_movielens20m_dssm.py` → `datasets/rec/movielens_20m_dssm` |
| Code commit | `dc5cb07` + the `user_fields` option (this run) |
| Device | Kunlunxin P800, card 7, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 2,745,158 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/dssm-ml20m/rec/ckpts/dssm_ml20m-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh rec dssm_ml20m 7` |

## Protocol

Same recipe as the [MovieLens-1M run](dssm_movielens_1m.md) (torch-rechub's
`run_ml_dssm.py`), with one structural change forced by the data:

| Setting | Value | Source |
| --- | --- | --- |
| dataset | MovieLens-20M: 20,000,263 ratings, 138,493 users, 26,745 items | `[code]` |
| user tower | **user_id + mean-pooled ≤50-item history only** — MovieLens-20M has no gender/age/occupation/zip | `[ours]` (the data has none) |
| item tower | movie_id + first genre | `[code]` |
| embedding / towers | 16 dims, [256, 128, 64] Linear-BN-PReLU, L2-normalised | `[code]` |
| loss / optimiser | point-wise BCE, Adam 1e-4, weight decay 1e-6 | `[code]` |
| batch / epochs / seed | 4096 / 10 / 2022 | `[code]` |
| train rows | 78,893,108 (1 positive + 3 popularity-sampled negatives per position) | `[code]` |
| test | 138,493 users, last item, full-catalogue retrieval over 26,745 items | `[code]` |

There is no published reference for this pair: torch-rechub reports 14.74%
Hit@100 on MovieLens-1M only (this repo reproduces 19.21% there).

## Curve (test)

| Epoch | Train loss | Hit@100 | NDCG@100 |
| --- | --- | --- | --- |
| 1 | 0.56681 | 0.1517 | 0.0334 |
| 2 | 0.56311 | 0.1631 | 0.0374 |
| 3 | 0.56308 | 0.1601 | — |
| 5 | 0.56309 | 0.1646 | 0.0384 |
| **6** | 0.56308 | 0.1662 | 0.0395 |
| 9 | 0.56308 | **0.1665** | — |
| 10 (last) | 0.56308 | 0.1647 | 0.0393 |

The curve is essentially flat from epoch 2 (train loss 0.5630 → 0.56308), i.e.
the model saturates at this capacity; the small differences between epochs are
noise.

## Results

| Metric | This run (epoch 10) | Best epoch (9) | MovieLens-1M run |
| --- | --- | --- | --- |
| Hit@100 = Recall@100 | 0.1647 | **0.1665** | 0.1921 |
| NDCG@100 | 0.0393 | — | 0.0430 |
| Hit@50 / NDCG@50 | 0.1001 / 0.0288 | — | 0.1127 / 0.0302 |
| Hit@10 / NDCG@10 | 0.0287 / 0.0135 | — | 0.0281 / 0.0121 |

Hit@100 is 0.027 lower than on MovieLens-1M, which is expected: the catalogue
is 7× larger (26,745 vs 3,706 items) and the user tower has no demographics.
The hit@10 rate is the same (0.0287 vs 0.0281).

Throughput: 183,777 samples/s (429 s/epoch over 78.9 M rows), 1 h 12 m total
including the 138 k-user full-ranking evaluation after every epoch.

## Notes

* **No demographics.**  MovieLens-20M ships ratings, movies and tags only.  The
  DSSM model now takes `model.user_fields` (default: torch-rechub's five
  fields), and this run sets `[user_id]`; the MovieLens-1M config is unchanged
  and was smoke-tested after the change.
* The 16 GB training table (78.9 M rows × 50-item history) and the 31 GB DIN
  table are the two largest artifacts of this experiment; both live under
  `/data2/jiangxuhui/datasets`.
* Evaluation is exact cosine top-k (torch-rechub uses an approximate Annoy
  index), so the ranking is at least as accurate as the reference's.

## Artifacts

| Path | Contents |
| --- | --- |
| `checkpoint/model.pt` | final weights (11 MB) |
| `checkpoint/history.json` | per-epoch train/test history |
| `checkpoint/metrics.json` | summary + full history |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python examples/rec/prepare_movielens20m_dssm.py
python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=dssm_ml20m train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
