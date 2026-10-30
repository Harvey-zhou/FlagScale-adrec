# LightGCN — Gowalla, paper protocol

| Field | Value |
| --- | --- |
| Status | **running** (started 2026-10-29 15:06:41 CST; epoch 267/1000 at 2026-10-30 09:48; 13 evaluations done) |
| Config | `examples/rec/conf/train/lightgcn_paper.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `da5195a` |
| Device | Kunlunxin P800, card 7, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 4,533,696 |
| Output | `/data2/jiangxuhui/logs/rec-paper-lightgcn/ckpts/lightgcn_paper-cuda/checkpoint/` |
| Command | `launch-rec-paper.sh 7 lightgcn_paper /data2/jiangxuhui/logs/rec-paper-lightgcn` |

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| task | graph collaborative filtering (BPR, pairwise) | — |
| dataset / split | `gowalla_lightgcn`: 29,858 users, 40,981 items, 810,128 train + 217,242 test interactions (= 1,027,370, the NGCF/LightGCN split); no validation split | `[paper]` Table 2 |
| optimizer | Adam, lr 1e-3 | `[paper]` 4.1.2 |
| warmup / schedule / grad clip | none / none / none | `[ours]` not in the paper or the code |
| Adam weight decay | 0 | `[paper]` |
| L2 on layer-0 embeddings | 1e-4 (Eq. 15 λ) | `[paper]` 4.5 (optimal for Gowalla) |
| batch size | 1,024 | `[paper]` 4.1.2 |
| embedding size | 64 | `[paper]` 4.1.2 |
| layers | 3, uniform layer combination 1/(K+1) | `[paper]` 4.1.2 / Table 3 |
| initialisation | Xavier (Glorot) | `[paper]` 4.1.2 (code uses N(0, 0.1)) |
| propagation | symmetric D^-1/2 A D^-1/2, no self-loops; computed with `index_select` + `index_add` | `[paper]` Eq. (3)/(7), `[ours]` kernel |
| epochs | 1,000 | `[paper]` 4.1.2 |
| evaluation | all-ranking recall@20 / ndcg@20, training items masked, every 20 epochs | `[paper]` 4.1 / Fig. 3 |
| early stopping | disabled | `[ours]` see below |
| seed | 2020 | `[code]` |

Paper target (Table 3, 3 layers): **recall@20 0.1823 / ndcg@20 0.1555**.
Released PyTorch code at 3 layers: 0.1824 / 0.1547.

### Why early stopping is off

The paper says the early-stopping and validation strategy is "the same as NGCF",
i.e. stop when validation recall@20 has not increased for 50 epochs (NGCF
§4.2.3).  The distributed Gowalla split has **no validation file** — the paper
received the train/test split from the NGCF authors — and the released code
trains the full 1,000 epochs.  Carving 10% out of `train.txt` would change the
training data and stop reproducing the published numbers, so the run follows the
code.  The trainer supports `early_stop_patience` if a validation split is added
later.

## Curve (test recall@20 / ndcg@20, every 20 epochs)

Evaluation happens every 20 epochs (the first row arrives after ~84 minutes).

| Epoch | recall@20 | ndcg@20 |
| --- | --- | --- |
| 20 | 0.1339 | 0.1160 |
| 40 | 0.1487 | 0.1282 |
| 60 | 0.1569 | 0.1345 |
| 80 | 0.1626 | 0.1397 |
| 100 | 0.1668 | 0.1431 |
| 120 | 0.1689 | 0.1449 |
| 140 | 0.1714 | 0.1466 |
| 160 | 0.1725 | 0.1476 |
| 180 | 0.1736 | 0.1489 |
| 200 | 0.1743 | 0.1496 |
| 220 | 0.1748 | 0.1495 |
| 240 | 0.1755 | 0.1504 |
| 260 | 0.1766 | 0.1511 |
| … | pending | pending |
| 1000 | pending | pending |

The curve is still climbing, with the gains per 20 epochs shrinking (0.0148 at
epoch 40, 0.0011 at epoch 260); the paper's 0.1823 comes from the full 1,000
epochs.

## Cost (measured on this host)

| Batch | Steps / epoch | Step time | Epoch | 1,000 epochs |
| --- | --- | --- | --- | --- |
| **1024 (paper, this run)** | 792 | 319 ms | 252.7 s | **~70 h** |
| 2048 (released code) | 396 | 319 ms | 126.4 s | ~35 h |

The per-step cost is dominated by the full-graph propagation (1,620,256 directed
edges × 64 dims × 3 layers: 159 ms forward + 160 ms backward) and does not
depend on the batch size, which is why the paper's smaller batch doubles the wall
clock.  `torch.sparse.mm` is not usable on this runtime (`CUDA error: invalid
device function`), so the paper's Eq. (7) is evaluated with the mathematically
identical `index_select` + `index_add` path.

## Results

Interim (epoch 260): recall@20 **0.1766** / ndcg@20 **0.1511** against the
paper's 0.1823 / 0.1555.  The metric is still rising at every evaluation, so this
is not a final number.  At 252 s/epoch the remaining 733 epochs are ~51 h, i.e.
completion around 2026-10-31 13:00 CST.  The final table (this run vs paper vs
released code) and the full curve will replace this section when the run
completes; per-epoch data is written to `history.json` after every evaluation.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | latest best evaluated epoch weights (rewritten on improvement) |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | written at the end: summary + full history |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=lightgcn_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
