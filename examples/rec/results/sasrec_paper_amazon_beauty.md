# SASRec — Amazon Beauty, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 275; best epoch 255) |
| Config | `examples/rec/conf/train/sasrec_beauty_paper.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `c068e12` (config); training code identical to `929e39a` |
| Started / finished | 2026-11-02 18:56:02 → 19:30:41 (CST) |
| Wall clock | 34 min 39 s (training 2,072.0 s) |
| Device | Kunlunxin P800, card 0, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 2,898,100 |
| Output | `/data2/jiangxuhui/logs/rec-beauty-paper/ckpts/sasrec_beauty_paper-cuda/checkpoint/` |
| Command | `launch-rec-paper.sh 0 sasrec_beauty_paper /data2/jiangxuhui/logs/rec-beauty-paper` |

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| task | sequential (next item) | — |
| dataset / split | `beauty_sasrec`: 52,204 users, 57,289 items, 394,908 interactions, 5-core, leave-one-out (`k=1..n-2` train, `n-1` validation, `n` test), max length 50, mean sequence 7.56 | `[paper]` IV-A/IV-C |
| optimizer | Adam, β1 0.9, β2 0.98 | `[paper]` IV-C / `[code]` (β2) |
| learning rate | 1e-3, constant | `[paper]` IV-C |
| warmup / schedule / grad clip | none / none / none | `[ours]` not in the paper or the code |
| weight decay / L2 | 0 | `[paper]`/`[code]` |
| batch size | 128 | `[paper]` IV-C |
| blocks / heads / d | 2 / 1 / 50 | `[paper]` IV-C, IV-F; `[code]` (d=50) |
| dropout | **0.5** | `[paper]` IV-C ("0.5 for the three sparser datasets") |
| positional embedding | learned, shared item embedding | `[paper]` III-A, IV-C |
| loss | BCE per non-padded position, one fresh negative per position per epoch | `[paper]` III-E |
| evaluation | HR@10 / NDCG@10 on 1 positive + 100 negatives, every epoch | `[paper]` IV-D |
| early stopping | stop after 20 epochs without validation NDCG@10 improvement, restore the best epoch | `[paper]` IV-B |
| epoch cap | 1,000 (safety; not reached) | `[ours]` |
| seed | 42 | `[code]` |

## Curve (validation; sampled, full per-epoch data in `history.json`)

381 steps per epoch at 6,575 samples/s (≈7.5 s/epoch).

| Epoch | Train loss | HR@10 | NDCG@10 |
| --- | --- | --- | --- |
| 1 | 1.2113 | 0.4340 | 0.2517 |
| 2 | 1.0637 | 0.4482 | 0.2667 |
| 3 | 1.0070 | 0.4572 | 0.2777 |
| 5 | 0.9298 | 0.4589 | 0.2783 |
| 10 | 0.7722 | 0.4491 | 0.2756 |
| 20 | 0.5827 | 0.4495 | 0.2824 |
| 30 | 0.4816 | 0.4573 | 0.2918 |
| 50 | 0.4048 | 0.4677 | 0.3024 |
| 75 | 0.3568 | 0.4741 | 0.3099 |
| 100 | 0.3271 | 0.4816 | 0.3159 |
| 150 | 0.2872 | 0.4898 | 0.3231 |
| 200 | 0.2643 | 0.4993 | 0.3319 |
| 250 | 0.2520 | 0.5018 | 0.3345 |
| **255 (best)** | **0.2532** | **0.5044** | **0.3372** |
| 275 (last) | 0.2509 | 0.5042 | 0.3366 |

Early stop fired at epoch 275: the best validation NDCG@10 was epoch 255 and
nothing improved for the next 20 epochs.  The best checkpoint was restored
before the test pass.  Validation NDCG@10 still creeps up (0.3345 → 0.3372
over epochs 250–255) but by less than the patience window allows; training loss
keeps falling, so there is no sign of the overfitting that the ML-1M run showed.

## Results

| Metric | This run | Paper (Table III) | Δ |
| --- | --- | --- | --- |
| HR@10 | 0.4604 | 0.4854 | −0.0250 |
| NDCG@10 | 0.3034 | 0.3219 | −0.0185 |
| HR@5 | 0.3598 | — | — |
| NDCG@5 | 0.2709 | — | — |

This is the largest gap of the four SASRec runs (ML-1M matched, Games/Steam are
0.7–1.6 points low).  The candidate causes are the same as in
[sasrec_paper_movielens_1m.md](sasrec_paper_movielens_1m.md) and none has been
isolated:

* the 100 evaluation negatives are drawn once with a fixed seed and stored,
  while the reference redraws them per evaluation;
* validation (second-to-last item) is 4.4 points above test (last item) here,
  versus 2.3 points on ML-1M — the last item is harder on Beauty, which is a
  data property, not a protocol difference.

**Do not read the gap as a defect of the paper settings.**  No setting above
deviates from `[paper]`; the only free choices are the epoch cap and the
implementation details listed in the protocol document.

## Notes

* Beauty is the sparsest of the four datasets (mean sequence 7.56), and its
  validation curve grows the slowest in absolute terms, yet it still needs
  ~250 epochs; the 1,000-epoch cap is not the binding constraint.
* Training throughput was 6,575 samples/s median, i.e. ~51 optimiser steps/s
  at batch 128 (381 steps per epoch) — the model is tiny (2.9 M parameters) and
  the loop is dominated by per-step overhead, as with the other rec runs.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation epoch (255) weights |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=sasrec_beauty_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
