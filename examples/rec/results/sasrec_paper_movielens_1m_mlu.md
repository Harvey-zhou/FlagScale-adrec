# SASRec — MovieLens-1M, paper protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 163; best epoch 143) |
| Started / finished | 2026-11-10 12:17:44 → 12:20:28 (CST) |
| Wall clock | 2 min 44 s (training 123.3 s) |
| Config | `examples/rec/conf/train/sasrec_paper.yaml` |
| Code commit | `d951e7b` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 1, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 211,950 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/rec/ckpts/sasrec_paper-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec sasrec_paper 1` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| task | sequential (next item) | — |
| dataset / split | `movielens_1m_sasrec`: 6,040 users, 3,416 items, 999,611 interactions, 5-core, leave-one-out (`k=1..n-2` train, `n-1` validation, `n` test), max length 200 | `[paper]` IV-A/IV-C |
| optimizer | Adam, β1 0.9, β2 0.98 | `[paper]` IV-C / `[code]` (β2) |
| learning rate | 1e-3, constant | `[paper]` IV-C |
| warmup / schedule / grad clip | none / none / none | `[ours]` not in the paper or the code |
| weight decay / L2 | 0 | `[paper]`/`[code]` |
| batch size | 128 | `[paper]` IV-C |
| blocks / heads / d | 2 / 1 / 50 | `[paper]` IV-C, IV-F; `[code]` (d=50) |
| dropout | 0.2 | `[paper]` IV-C (ML-1M) |
| positional embedding | learned, shared item embedding | `[paper]` III-A, IV-C |
| loss | BCE per non-padded position, one fresh negative per position per epoch | `[paper]` III-E |
| evaluation | HR@10 / NDCG@10 on 1 positive + 100 negatives, every epoch | `[paper]` IV-D |
| early stopping | stop after 20 epochs without validation NDCG@10 improvement, restore the best epoch | `[paper]` IV-B |
| epoch cap | 1,000 (safety) | `[ours]` |
| seed | 42 | `[code]` |

## Curve (validation; sampled, full per-epoch data in `history.json`)

| Epoch | Train loss | HR@10 | NDCG@10 |
| --- | --- | --- | --- |
| 1 | 1.1691 | 0.4320 | 0.2355 |
| 5 | 0.8733 | 0.5715 | 0.3302 |
| 10 | 0.6860 | 0.7280 | 0.4622 |
| 20 | 0.5643 | 0.7916 | 0.5311 |
| 30 | 0.5102 | 0.8142 | 0.5674 |
| 40 | 0.4833 | 0.8260 | 0.5842 |
| 60 | 0.4539 | 0.8381 | 0.6014 |
| 80 | 0.4383 | 0.8401 | 0.6084 |
| 100 | 0.4265 | 0.8421 | 0.6110 |
| 120 | 0.4178 | 0.8449 | 0.6150 |
| 140 | 0.4134 | 0.8472 | 0.6149 |
| **143 (best)** | **0.4123** | **0.8469** | **0.6176** |
| 150 | 0.4128 | 0.8457 | 0.6136 |
| 159 | 0.4089 | 0.8454 | 0.6166 |
| 163 (last) | 0.4088 | 0.8449 | 0.6127 |

Early stop fired at epoch 163: the best validation NDCG@10 was epoch 143 and
nothing improved for the next 20 epochs.  The best checkpoint was restored
before the test pass.

## Results

| Metric | This run | P800 same config | Paper (Table III) |
| --- | --- | --- | --- |
| HR@10 | 0.8174 | 0.8204 | 0.8245 |
| NDCG@10 | **0.5895** | **0.5910** | **0.5905** |
| HR@5 | 0.7179 | 0.7169 | — |
| NDCG@5 | 0.5573 | 0.5575 | — |

The MLU test metric lands 0.0030 HR@10 / 0.0015 NDCG@10 under the P800 run and
0.0071 / 0.0010 under the paper.  The P800 row is the run recorded in
[`sasrec_paper_movielens_1m.md`](sasrec_paper_movielens_1m.md), launched at code
commit `da5195a`, which is not part of this branch's history (see the
comparability note in [../../CAMBRICON.md](../../CAMBRICON.md)); treat it as a
reference rather than a strict like-for-like.  The validation-to-test gap
(0.8469 → 0.8174 HR@10) is the same effect the P800 page reports.

## Notes

* Throughput: **8,710 samples/s median** over the 163 epochs (fastest 9,426,
  slowest 8,062); training took 123.3 s and the final test pass 0.04 s.  Card 1
  was exclusive for this run.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; the clock was not sampled while training.
* Evaluation uses the dataset's fixed 100 negatives, which exclude all of the
  user's items; the reference redraws negatives every evaluation (the same
  caveat as the P800 page).

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation epoch (143) weights, 839 KB |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh rec sasrec_paper 1
```
