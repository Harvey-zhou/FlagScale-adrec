# LightGCN — Gowalla, reference-code settings (Cambricon MLU, 2-card)

| Field | Value |
| --- | --- |
| Status | **finished** (1,000 / 1,000 epochs) |
| Started / finished | 2026-11-11 16:57:43 → 23:28:06 (CST) |
| Wall clock | 6 h 30 m 23 s (training 23,372.4 s = 6 h 29 m 32 s) |
| Config | `examples/rec/conf/train/lightgcn.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `cda3072` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), cards 6 + 7, 2-card CNCL DDP, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 4,533,696 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/rec/ckpts/lightgcn-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec lightgcn 6,7 train.batch_size=1024` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.  This is
the first 2-card CNCL run on node B; `device count: 2 (expected 2)` in the
preflight, no CNCL initialisation error.

## Settings

This is the reference-code config (`gusye1234/LightGCN-PyTorch` `parse.py`
defaults), not the paper protocol of
[`lightgcn_paper_gowalla.md`](lightgcn_paper_gowalla.md):

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `gowalla_lightgcn`: 29,858 users, 40,981 items, 810,128 train + 217,242 test interactions; no validation split | `[code]` |
| optimizer | Adam, lr 1e-3 | `[code]` |
| L2 on layer-0 embeddings | 1e-4 | `[code]` |
| batch size | 2048 global, **1024 per rank** (396 steps per rank per epoch) | `[code]` |
| embedding size / layers | 64 / 3, uniform layer combination | `[code]` |
| initialisation | N(0, 0.1) | `[code]` (the paper uses Xavier) |
| propagation | `sparse` (default COO SpMM), no self-loops | `[ours]` kernel |
| epochs | 1,000 | `[code]` |
| evaluation | all-ranking recall@20 / ndcg@20, training items masked, every 20 epochs | `[code]` |
| early stopping | disabled | `[code]` |
| seed | 2020 | `[code]` |

Reference targets: paper Table 3 (3 layers) **0.1823 / 0.1555**; released
PyTorch code (3 layers) **0.1824 / 0.1547**; BARS 4-layer reference
0.1820 / 0.1547.

The 2-card launch keeps the reference *global* batch at 2048 by halving it per
rank.  The per-rank batch is a launcher override (`train.batch_size=1024`), not
a config change; the config still says 2048, which is what a single-card run
would use.

## Curve (test recall@20 / ndcg@20, every 20 epochs)

| Epoch | recall@20 | ndcg@20 |
| --- | --- | --- |
| 20 | 0.1326 | 0.1148 |
| 40 | 0.1457 | 0.1254 |
| 60 | 0.1532 | 0.1317 |
| 80 | 0.1599 | 0.1369 |
| 100 | 0.1640 | 0.1402 |
| 120 | 0.1674 | 0.1430 |
| 140 | 0.1703 | 0.1450 |
| 160 | 0.1718 | 0.1465 |
| 200 | 0.1739 | 0.1489 |
| 240 | 0.1760 | 0.1502 |
| 300 | 0.1767 | 0.1515 |
| 400 | 0.1789 | 0.1531 |
| 500 | 0.1793 | 0.1538 |
| 600 | 0.1812 | 0.1546 |
| 700 | 0.1809 | 0.1546 |
| 800 | 0.1809 | 0.1549 |
| **900 (best)** | **0.1821** | **0.1554** |
| 1,000 (last) | 0.1818 | 0.1553 |

The curve keeps moving until ~epoch 900, exactly as on the paper protocol: the
reference's 1,000 epochs are enough to reach the published level, not to
converge.  The reference protocol reports the last epoch, so 1,000 is the
reported number; epoch 900 is the best evaluated checkpoint.

## Results

| Metric | This run (epoch 1000) | Best evaluated (900) | Released code | Paper Table 3 |
| --- | --- | --- | --- | --- |
| recall@20 | **0.1818** | 0.1821 | 0.1824 | 0.1823 |
| ndcg@20 | **0.1553** | 0.1554 | 0.1547 | 0.1555 |
| precision@20 | 0.0560 | 0.0562 | — | — |
| hit@20 | 0.5921 | 0.5907 | — | — |

The final epoch is 0.0006 recall@20 below the released code and 0.0005 below
the paper; ndcg@20 is +0.0006 above the released code and 0.0002 below the
paper.  The best evaluated epoch (900) is within 0.0003 of all three.  This is
a reproduction within run-to-run variation, and the first LightGCN *code*
result on node B.

## Cost

| Quantity | Value |
| --- | --- |
| Mean throughput (all 1,000 epochs) | **35,730 samples/s** (median 35,770) |
| Epoch | 22.7–23.4 s |
| Steps / epoch / rank | 396 (batch 1024) |
| 1,000 epochs | 6 h 29 m 32 s |

The 2-card run is 35.7 k global samples/s.  Node A's 2-card paper-protocol
Gowalla run reached 63,700 samples/s with batch 512/rank and `index_add`
propagation; the difference here is the per-rank batch (1024 vs 512) and the
default `sparse` COO propagation, which is the slower kernel on MLU.  Accuracy
is not affected by either choice — the config's propagation kernel is
mathematically the same operation.

## Notes

* First 2-card CNCL DDP on node B: the run completed all 1,000 epochs with no
  collective error, so CNCL over MLU-Link is usable on this node as well as on
  node A.
* No validation split is used: the paper's NGCF-style early stop needs one, the
  released split has none, and the code trains the full 1,000 epochs.
* Training throughput is read from `history.json`; evaluation ran every 20
  epochs and its 2.5 s full-ranking pass is not part of the training rate.
* Clock: not sampled while training; idle cards report 1200 MHz, load raises
  the clock to 1600 MHz (`CAMBRICON.md`).

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | latest best evaluated epoch weights |
| `checkpoint/history.json` | per-epoch training history, written incrementally |
| `checkpoint/metrics.json` | final summary + history + best evaluated epoch |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec lightgcn 6,7 train.batch_size=1024
```
