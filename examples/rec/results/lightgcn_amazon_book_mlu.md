# LightGCN — Amazon-Book, reference-code settings (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (1,000 / 1,000 epochs) |
| Started / finished | 2026-11-09 23:00:35 → 2026-11-13 08:04:06 (CST) |
| Wall clock | 81 h 03 m 31 s (training 291,762.8 s = 81 h 02 m 43 s) |
| Config | `examples/rec/conf/train/lightgcn_amazon_book.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `d951e7b` (working tree clean at launch) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 0, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 9,231,488 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/rec/ckpts/lightgcn_amazon_book-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec lightgcn_amazon_book 0` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

Same recipe as the Gowalla code-settings run, on the authors' Amazon-Book split:

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `amazon_book_lightgcn`: 52,643 users, 91,599 items, 2,380,730 train + 603,378 test interactions (= 2,984,108); no validation split | `[paper]` Table 2 |
| optimizer | Adam, lr 1e-3 | `[code]` |
| L2 on layer-0 embeddings | 1e-4 | `[code]` |
| batch size | 2048 | `[code]` / `[paper]` |
| embedding size / layers | 64 / 3, uniform layer combination | `[code]` |
| initialisation | N(0, 0.1) | `[code]` |
| propagation | `sparse` COO SpMM, no self-loops | `[ours]` kernel |
| epochs | 1,000 | `[code]` |
| evaluation | all-ranking recall@20 / ndcg@20, training items masked, every 20 epochs | `[code]` |
| early stopping | disabled (no validation split) | `[code]` |
| seed | 2020 | `[code]` |

Paper target (Table 3, 3 layers): **recall@20 0.0410 / ndcg@20 0.0318**.  The
released PyTorch README does not list Amazon-Book.

## Curve (test recall@20 / ndcg@20, every 20 epochs)

| Epoch | recall@20 | ndcg@20 |
| --- | --- | --- |
| 100 | 0.0371 | 0.0288 |
| 200 | 0.0397 | 0.0308 |
| 300 | 0.0406 | 0.0314 |
| 400 | 0.0412 | 0.0319 |
| 500 | 0.0412 | 0.0318 |
| 600 | 0.0410 | 0.0318 |
| 700 | 0.0412 | 0.0320 |
| **800 (best)** | **0.0417** | **0.0322** |
| 900 | 0.0413 | 0.0321 |
| 1,000 (last) | 0.0415 | 0.0322 |

The curve plateaus around epoch 400 and oscillates within ~0.0007 recall after
that; the best evaluated epoch is 800.  As on Gowalla, 1,000 epochs are enough
to reach the published level, not to converge.

## Results

| Metric | This run (epoch 1000) | Best evaluated (800) | Paper Table 3 |
| --- | --- | --- | --- |
| recall@20 | **0.041545** | 0.041707 | 0.0410 |
| ndcg@20 | **0.032223** | 0.032249 | 0.0318 |
| precision@20 | 0.017182 | 0.017222 | — |
| hit@20 | 0.245597 | 0.245768 | — |

The final epoch is +0.0005 recall@20 / +0.0004 ndcg@20 above the paper, and the
best evaluated epoch is +0.0007 / +0.0004 above.  This reproduces the paper
within its own run-to-run spread on a second dataset, and is the first
Amazon-Book LightGCN run on MLU.

## Cost

| Quantity | Value |
| --- | --- |
| Mean throughput (all 1,000 epochs) | **8,295 samples/s** (median 8,299) |
| Epoch | ~291.8 s (4 min 52 s) |
| 1,000 epochs | 81 h 02 m 43 s |

The cost is ~2.9× the Gowalla run per epoch (9.2 M parameters, 2.98 M
interactions, and a 91,599-item score matrix per eval batch).  The per-epoch
rate was flat at 8.3 k samples/s, i.e. card 0 was not shared.

## Notes

* First complete multi-day MLU run on node B: one process, no restart, no CNCL
  involvement (single card).
* No validation split is used; the paper's NGCF-style early stop needs one and
  the released split has none, so the code protocol (last epoch) is reported
  alongside the best evaluated epoch.
* Evaluation is expensive here — the score matrix is [4096, 91599] per user
  batch (~1.5 GB in fp32) — but each eval pass is still only ~5.1 s and is not
  part of the training throughput.
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

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec lightgcn_amazon_book 0
```
