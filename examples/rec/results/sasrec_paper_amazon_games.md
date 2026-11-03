# SASRec — Amazon Games, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 182; best epoch 162) |
| Config | `examples/rec/conf/train/sasrec_games_paper.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `c068e12` (config); training code identical to `929e39a` |
| Started / finished | 2026-11-02 19:31:41 → 19:47:02 (CST) |
| Wall clock | 15 min 21 s (training 914.3 s) |
| Device | Kunlunxin P800, card 0, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 1,219,400 |
| Output | `/data2/jiangxuhui/logs/rec-games-paper/ckpts/sasrec_games_paper-cuda/checkpoint/` |
| Command | `launch-rec-paper.sh 0 sasrec_games_paper /data2/jiangxuhui/logs/rec-games-paper` |

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| task | sequential (next item) | — |
| dataset / split | `games_sasrec`: 31,013 users, 23,715 items, 287,107 interactions, 5-core, leave-one-out (`k=1..n-2` train, `n-1` validation, `n` test), max length 50, mean sequence 9.26 | `[paper]` IV-A/IV-C |
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

240 steps per epoch at 6,362 samples/s (≈5.0 s/epoch).

| Epoch | Train loss | HR@10 | NDCG@10 |
| --- | --- | --- | --- |
| 1 | 1.1999 | 0.5002 | 0.2954 |
| 2 | 1.0332 | 0.5069 | 0.3014 |
| 3 | 0.9872 | 0.5275 | 0.3170 |
| 5 | 0.8958 | 0.5851 | 0.3623 |
| 10 | 0.7353 | 0.6556 | 0.4250 |
| 20 | 0.5494 | 0.7027 | 0.4740 |
| 30 | 0.4517 | 0.7206 | 0.4979 |
| 50 | 0.3614 | 0.7442 | 0.5293 |
| 75 | 0.3146 | 0.7507 | 0.5401 |
| 100 | 0.2990 | 0.7567 | 0.5459 |
| 150 | 0.2758 | 0.7602 | 0.5511 |
| **162 (best)** | **0.2694** | **0.7634** | **0.5551** |
| 182 (last) | 0.2675 | 0.7647 | 0.5541 |

Early stop fired at epoch 182: the best validation NDCG@10 was epoch 162 and
nothing improved for the next 20 epochs.  The best checkpoint was restored
before the test pass.  The curve flattens from epoch ~75 (NDCG@10 0.5401) and
gains only 0.015 over the next 100 epochs.

## Results

| Metric | This run | Paper (Table III) | Δ |
| --- | --- | --- | --- |
| HR@10 | 0.7340 | 0.7410 | −0.0070 |
| NDCG@10 | 0.5281 | 0.5360 | −0.0079 |
| HR@5 | 0.6328 | — | — |
| NDCG@5 | 0.4953 | — | — |

The closest of the three sparse-set runs (0.7–0.8 points below the paper).  The
same two caveats as [sasrec_paper_movielens_1m.md](sasrec_paper_movielens_1m.md)
apply: fixed 100 evaluation negatives per user instead of redrawing them, and
the implementation's tie-breaking.  Neither has been isolated, and no
paper-specified setting differs here.

## Notes

* 31,013 users and only 287 k interactions: the smallest graph of the four runs,
  so it converges fastest (best epoch 162, 15 min wall clock).
* Validation HR@10 0.7634 is 2.9 points above the test HR@10 0.7340, in the same
  range as ML-1M (2.3 points).

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation epoch (162) weights |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=sasrec_games_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
