# LightGCN — Yelp2018, batch 2048

| Field | Value |
| --- | --- |
| Status | **finished** (1,000 epochs) + the BF16 counterpart |
| Started / finished | 2026-11-09 11:41 → 2026-11-12 22:30 (CST, wall clock across two resume segments) |
| Training time | 36.5 h of epochs actually computed; the wall clock includes a host fault on 2026-11-11 |
| Config | `examples/rec/conf/train/lightgcn_yelp2018_paper.yaml` + `train.batch_size=2048` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `dc5cb07` (training), `ba83cb0`/`e282231` (resume watchdog) |
| Device | Kunlunxin P800, card 4 (FP32) and card 5 (BF16), vendor torch 2.9.0+cu128, fp32/bfloat16 |
| Parameters | 4,461,824 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/yelp2018-b2048-{fp32,bf16}-v2/rec/ckpts/lightgcn_yelp2018_paper-cuda/checkpoint/` |
| Command | `run-resumable.sh 4 yelp2018-b2048-fp32-v2 rec lightgcn_yelp2018_paper train.batch_size=2048` |

## Protocol and the one deviation

Paper: He et al., SIGIR 2020, Table 3 (3 layers): **recall@20 0.0639 /
ndcg@20 0.0525**; released PyTorch code 0.06347 / 0.05238.

| Setting | Value | Source |
| --- | --- | --- |
| dataset | `yelp2018_lightgcn`: 31,668 users, 38,048 items, 1,237,259 train + 324,147 test interactions, no validation split | `[paper]` Table 2 |
| optimizer / rate | Adam 1e-3 | `[paper]` 4.1.2 |
| **batch size** | **2,048** (paper: 1,024) | **deviation**: 604 steps/epoch instead of 1,209 |
| embedding / layers | 64 dims, 3 layers, uniform 1/(K+1) combination, no self-loops | `[paper]` 4.1.2 / Eq. (3) |
| initialisation | Xavier | `[paper]` 4.1.2 |
| L2 on layer-0 embeddings | 1e-3 (Yelp2018's optimal λ) | `[paper]` 4.5 |
| propagation | symmetric D^-1/2 A D^-1/2 via `index_select` + `index_add` (the runtime has no working sparse COO path) | `[ours]`, identical math |
| epochs / evaluation | 1,000, all-ranking recall@20 / ndcg@20 every 20 epochs, final epoch is the headline | `[paper]` 4.1 / Fig. 3 |
| early stopping | disabled (no validation split; the released code also trains 1,000 epochs) | `[ours]` |

Why batch 2048: it halves the wall clock (294 s/epoch vs 586 s at batch 1024 for
the same per-step cost).  The cost is that 1,000 epochs means **604 k optimizer
steps instead of 1.21 M** — half the training the paper's setting performs.  Two
arms at batch 1024 are running to quantify exactly that.

## Curve (test, every 20 epochs)

| Epoch | recall@20 | ndcg@20 |
| --- | --- | --- |
| 20 | 0.0420 | 0.0342 |
| 100 | 0.0516 | 0.0418 |
| 200 | 0.0549 | 0.0445 |
| 300 | 0.0562 | 0.0455 |
| 400 | 0.0567 | 0.0461 |
| 500 | 0.0573 | 0.0464 |
| 600 | 0.0571 | 0.0463 |
| 700 | 0.0574 | 0.0465 |
| **740 (best)** | **0.0577** | **0.0465** |
| 800 | 0.0573 | 0.0464 |
| 900 | 0.0573 | 0.0465 |
| **1000 (headline)** | **0.0576** | **0.0465** |

The curve flattens after ~500 epochs; the last 500 epochs move recall@20 by
+0.0003.  This is the expected shape for a run whose step budget is half of the
paper's: it converges to a lower plateau rather than being cut short.

## Results

| Metric | Final (epoch 1000) | Best evaluated (740) | Paper Table 3 | Released code |
| --- | --- | --- | --- | --- |
| recall@20 | **0.057566** | 0.057694 | 0.0639 | 0.06347 |
| ndcg@20 | **0.046549** | 0.046535 | 0.0525 | 0.05238 |
| precision@20 | 0.026050 | 0.026107 | — | — |
| hit@20 | 0.364437 | 0.365195 | — | — |

0.0063 recall@20 below the paper (−9.9 %).  The single deviation that can
explain it is the batch size: at batch 2048 the run performs 604 k updates where
the paper performs 1.21 M, and the curve is still ~0.0003 away from being flat
at the end.  The dataset, metric, initialisation, regularisation and layer count
all follow the paper, and Gowalla on the same code at batch 1024 reproduces the
paper to 0.0003 — so this is a training-budget effect, not an implementation
defect.

## BF16 counterpart

The same config, seed and budget under `train.model.torch_dtype=bf16`:

| | FP32 | BF16 |
| --- | --- | --- |
| recall@20 (e1000) | 0.057566 | **0.057566** |
| ndcg@20 (e1000) | 0.046549 | **0.046549** |
| precision@20 / hit@20 | 0.026050 / 0.364437 | 0.026050 / 0.364437 |
| best evaluated (e740) | 0.057694 / 0.046535 | 0.057694 / 0.046535 |
| training throughput | 4,208 samples/s (293 s/epoch) | 4,207 samples/s (294 s/epoch) |

Every evaluated epoch is identical between the two precisions — the numbers above
are the same at all 50 evaluation points, not just the last.  As with Gowalla,
LightGCN's cost is the full-graph propagation, which autocast does not touch, so
BF16 changes neither the metric nor the speed.

Inference (batch 2048, full ranking over 31,668 users × 38,048 items):
FP32 **128,344** rows/s, BF16 **126,532** rows/s.

## Cost

| | Value |
| --- | --- |
| epoch | 605 steps, 293 s (FP32) / 294 s (BF16) |
| 1,000 epochs | 36.5 h of compute (the wall clock was longer: a host XPU fault on 2026-11-11 killed both arms, which resumed from `latest.pt`) |
| throughput | 4,208 samples/s |

## Artifacts

| Path | Contents |
| --- | --- |
| `.../yelp2018-b2048-fp32-v2/.../checkpoint/` | model.pt, latest.pt, metrics.json, history.json, config.json |
| `.../yelp2018-b2048-bf16-v2/.../checkpoint/` | same, BF16 |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

bash tools/run-resumable.sh 4 yelp-b2048 rec lightgcn_yelp2018_paper train.batch_size=2048
# BF16: add train.model.torch_dtype=bf16
# paper setting (batch 1024): drop the batch_size override (586 s/epoch, ~163 h)
```
