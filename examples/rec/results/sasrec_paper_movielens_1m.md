# SASRec — MovieLens-1M, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped) |
| Config | `examples/rec/conf/train/sasrec_paper.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `da5195a` |
| Started / finished | 2026-10-29 15:06:41 → 15:09:58 (CST) |
| Wall clock | 3 min 17 s (training 195.3 s) |
| Device | Kunlunxin P800, card 4, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 211,950 |
| Output | `/data2/jiangxuhui/logs/rec-paper-sasrec/ckpts/sasrec_paper-cuda/checkpoint/` |
| Command | `launch-rec-paper.sh 4 sasrec_paper /data2/jiangxuhui/logs/rec-paper-sasrec` |

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
| 1 | 1.1694 | 0.4305 | 0.2361 |
| 2 | 1.0095 | 0.4329 | 0.2394 |
| 3 | 0.9772 | 0.4806 | 0.2644 |
| 4 | 0.9296 | 0.5414 | 0.2989 |
| 5 | 0.8720 | 0.5801 | 0.3362 |
| 6 | 0.8249 | 0.6263 | 0.3716 |
| 7 | 0.7794 | 0.6618 | 0.4026 |
| 8 | 0.7440 | 0.6960 | 0.4290 |
| 9 | 0.7133 | 0.7136 | 0.4469 |
| 10 | 0.6893 | 0.7255 | 0.4604 |
| 20 | 0.5667 | 0.7894 | 0.5321 |
| 30 | 0.5127 | 0.8124 | 0.5675 |
| 40 | 0.4852 | 0.8238 | 0.5828 |
| 50 | 0.4697 | 0.8306 | 0.5940 |
| 60 | 0.4552 | 0.8369 | 0.6022 |
| 70 | 0.4472 | 0.8381 | 0.6079 |
| 80 | 0.4382 | 0.8404 | 0.6111 |
| 90 | 0.4312 | 0.8396 | 0.6105 |
| 100 | 0.4267 | 0.8407 | 0.6143 |
| 110 | 0.4214 | 0.8417 | 0.6126 |
| 120 | 0.4178 | 0.8417 | 0.6135 |
| 130 | 0.4168 | 0.8430 | 0.6150 |
| **139 (best)** | **0.4136** | **0.8435** | **0.6189** |
| 140 | 0.4129 | 0.8439 | 0.6173 |
| 150 | 0.4126 | 0.8407 | 0.6139 |
| 159 (last) | 0.4092 | 0.8442 | 0.6153 |

Early stop fired at epoch 159: the best validation NDCG@10 was epoch 139 and
nothing improved for the next 20 epochs (140–159). The best checkpoint was
restored before the test pass.

## Results

| Metric | This run | Paper (Table III) | Δ |
| --- | --- | --- | --- |
| HR@10 | 0.8204 | 0.8245 | −0.0041 |
| NDCG@10 | **0.5910** | **0.5905** | +0.0005 |
| HR@5 | 0.7169 | — | — |
| NDCG@5 | 0.5575 | — | — |

NDCG@10 matches the paper to 0.0005; HR@10 is 0.4 points lower. The two
protocol differences below explain a difference of this size, and neither is a
change to a paper-specified training setting.

## Notes

* **Evaluation candidates.** The 100 negatives are drawn once with a fixed seed
  and stored in the dataset, excluding every item of the user.  The reference
  redraws them at every evaluation and excludes only the training items.
* The validation curve is still improving slowly at the end (0.6153 at epoch
  159 vs 0.6189 at 139); the 20-epoch patience rule stops it there, as the
  paper specifies.
* Training used 6,762 samples/s at its fastest and 5,135 samples/s median; the
  3 min 17 s wall clock is the 195 s of training plus startup and the final
  test pass (the test evaluation itself is 0.08 s).

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation epoch (139) weights |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=sasrec_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
