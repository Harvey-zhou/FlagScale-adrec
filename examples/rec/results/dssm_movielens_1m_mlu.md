# DSSM (two-tower) — MovieLens-1M, torch-rechub protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (10 / 10 epochs) |
| Started / finished | 2026-11-10 12:20:28 → 12:23:47 (CST) |
| Wall clock | 3 min 19 s (training 137.5 s) |
| Config | `examples/rec/conf/train/dssm.yaml` |
| Code commit | `d951e7b` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 1, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 329,222 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/rec/ckpts/dssm-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec dssm 1` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; no CPU fallback and no missing MLU
kernel in the launcher's log scan.  This run followed the SASRec run on the same
card, so card 1 was otherwise idle.

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| task | two-tower retrieval, point-wise BCE with 3 popularity-sampled negatives | `[framework]` torch-rechub |
| dataset / split | `movielens_1m_dssm`: 6,040 users, 3,706 items, 3,952,516 train rows; last item per user as test (6,040 users), full-catalogue retrieval | `[framework]` |
| user tower | user_id, gender, age, occupation, zip, mean-pooled history ≤ 50 | `[framework]` |
| item tower | movie_id, first genre | `[framework]` |
| embedding size | 16 | `[framework]` |
| tower hidden | [256, 128, 64], BatchNorm + PReLU, L2-normalized output | `[framework]` |
| temperature | 1.0 (torch-rechub passes 0.02 but never applies it in `DSSM.forward`) | `[framework]` |
| optimizer | Adam, lr 1e-4, weight decay 1e-6 | `[framework]` |
| batch size | 4096 | `[framework]` |
| epochs / evaluation | 10, test evaluated every epoch, select hit@100 | `[framework]` |
| seed | 2022 | `[framework]` |

## Curve (test each epoch; there is no validation split)

| Epoch | Train loss | Hit@100 = Recall@100 | NDCG@100 |
| --- | --- | --- | --- |
| 1 | 0.6050 | 0.0387 | 0.0088 |
| 2 | 0.5697 | 0.0475 | 0.0100 |
| 3 | 0.5640 | 0.0507 | 0.0102 |
| 4 | 0.5601 | 0.0508 | 0.0103 |
| 5 | 0.5571 | 0.1344 | 0.0257 |
| 6 | 0.5552 | 0.1455 | 0.0285 |
| 7 | 0.5541 | 0.1546 | 0.0310 |
| 8 | 0.5533 | 0.1558 | 0.0314 |
| 9 | 0.5526 | 0.1576 | 0.0328 |
| **10 (best / last)** | **0.5520** | **0.1599** | **0.0334** |

The metric jumps at epoch 5 and then improves slowly to the end, so the fixed
10-epoch budget truncates a curve that has not flattened; no early stop is used,
matching the torch-rechub script.

## Results

| Metric | This run | torch-rechub reference |
| --- | --- | --- |
| Hit@100 = Recall@100 | **0.1599** | 0.1474 |
| NDCG@100 | 0.0334 | — |
| Recall@10 | 0.0152 | — |
| NDCG@10 | 0.0064 | — |

The MLU run is 1.25 points above the torch-rechub README's 14.74% Hit@100.
The reference is loosely specified (see [../EVIDENCE.md](../EVIDENCE.md)): the
script prints top-10 while the README reports @100, `temperature` is passed but
not applied, and the candidate/negative construction can differ.  The gap is
therefore recorded, not read as a model improvement.

## Notes

* Throughput: **303,033 samples/s median** (12.8–13.5 s per epoch for 965 steps
  of 4096); the final test pass took 0.66 s.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; not sampled during training.
* Test evaluation is exhaustive over the 3,706-item catalogue, so the 100-cut
  metrics are all-ranking metrics on the held-out last item.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best hit@100 epoch (10) weights, 1.3 MB |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec dssm 1
```
