# SASRec — Steam, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 78; best epoch 58) |
| Config | `examples/rec/conf/train/sasrec_steam_paper.yaml` |
| Protocol document | [../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md) |
| Code commit | `c068e12` (config); training code identical to `929e39a` |
| Started / finished | 2026-11-02 19:48:02 → 21:06:37 (CST) |
| Wall clock | 1 h 18 m (training 4,702.3 s) |
| Device | Kunlunxin P800, card 0, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 686,000 |
| Output | `/data2/jiangxuhui/logs/rec-steam-paper/ckpts/sasrec_steam_paper-cuda/checkpoint/` |
| Command | `launch-rec-paper.sh 0 sasrec_steam_paper /data2/jiangxuhui/logs/rec-steam-paper` |

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| task | sequential (next item) | — |
| dataset / split | `steam_sasrec`: 334,730 users, 13,047 items, 3,686,172 interactions, 5-core, leave-one-out (`k=1..n-2` train, `n-1` validation, `n` test), max length 50, mean sequence 11.01 | `[paper]` IV-A/IV-C |
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

2,519 steps per epoch (322 k training rows) at 6,279 samples/s (≈60 s/epoch).

| Epoch | Train loss | HR@10 | NDCG@10 |
| --- | --- | --- | --- |
| 1 | 0.7070 | 0.8045 | 0.5390 |
| 2 | 0.6267 | 0.8190 | 0.5578 |
| 3 | 0.6072 | 0.8278 | 0.5680 |
| 5 | 0.5846 | 0.8429 | 0.5859 |
| 10 | 0.5615 | 0.8531 | 0.6011 |
| 20 | 0.5465 | 0.8606 | 0.6107 |
| 30 | 0.5417 | 0.8628 | 0.6138 |
| 50 | 0.5368 | 0.8645 | 0.6161 |
| **58 (best)** | **0.5355** | **0.8657** | **0.6184** |
| 75 | 0.5342 | 0.8659 | 0.6177 |
| 78 (last) | 0.5344 | 0.8649 | 0.6159 |

Early stop fired at epoch 78: the best validation NDCG@10 was epoch 58 and
nothing improved for the next 20 epochs.  The best checkpoint was restored
before the test pass.  The model saturates quickly — by epoch 10 it is already
at NDCG@10 0.6011 — which is consistent with the dense item statistics (3.7 M
interactions over only 13 k items, ≈283 interactions per item).

## Results

| Metric | This run | Paper (Table III) | Δ |
| --- | --- | --- | --- |
| HR@10 | 0.8589 | 0.8729 | −0.0140 |
| NDCG@10 | 0.6150 | 0.6306 | −0.0156 |
| HR@5 | 0.7429 | — | — |
| NDCG@5 | 0.5772 | — | — |

Same caveats as the other runs in this family (fixed 100 evaluation negatives,
tie-breaking); no paper-specified training setting differs.  Unlike ML-1M and
Beauty, the validation→test drop is small here (0.8657 → 0.8589, 0.7 points),
which is consistent with many items having a large interaction count: the last
item is not systematically much rarer than the second-to-last.

## Notes

* Steam is by far the largest of the four SASRec datasets (3.7 M interactions,
  335 k users) but the item vocabulary is the smallest (13,047), so the model
  has only 686 k parameters and one epoch costs ~60 s.
* Throughput is 6,279 samples/s median; the 1 h 18 m wall clock is dominated by
  the 2,519 steps per epoch, not by model size.

## BF16 counterpart

The same config, seed and epoch budget under `train.model.torch_dtype=bf16`
(autocast with fp32 master weights), run on the P800:

| | FP32 | BF16 | Δ |
| --- | --- | --- | --- |
| HR@10 | 0.8589 | **0.8593** | +0.0004 |
| NDCG@10 | 0.6150 | **0.6152** | +0.0002 |
| epochs run (best) | 78 | 78 (58) | — |
| training throughput | 6,271 samples/s | 13,525 samples/s | +116% |
| inference (rows/s, batch 1024) | 268,304 | 216,555 | — |

The metric difference is inside run-to-run variation; on Beauty the BF16 run
simply early-stopped later (78 epochs against 78), which is what
moves the number, not bf16 arithmetic.  Inference is ~20 % slower in bf16 here,
much less than MovieLens-1M's 66 %, because these sequences are 50 items long
instead of 200.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation epoch (58) weights |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=sasrec_steam_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
