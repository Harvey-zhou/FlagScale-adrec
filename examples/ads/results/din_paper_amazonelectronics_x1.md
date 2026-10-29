# DIN — Amazon Electronics_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **running** (started 2026-10-29 10:06:07 CST; 16/50 epochs done at 15:08) |
| Config | `examples/ads/conf/train/din_paper.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | `8211572` |
| Device | Kunlunxin P800, card 6, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 4,152,226 |
| Output | `/data2/jiangxuhui/logs/ads-paper-queue/ckpts/din_paper-cuda/checkpoint/` |
| Command | `launch-ads-paper.sh 6 din_paper /data2/jiangxuhui/logs/ads-paper-queue` |

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `amazonelectronics_x1`: 192,403 users, 63,003 item ids, 803 category ids; train 2,608,764 rows (1,304,382 pos + 1,304,382 neg), test 384,806 rows (192,403 pos + neg); train `k=1..n-2`, test the last item | `[paper]` §6.1 |
| optimizer | SGD, momentum 0 | `[paper]` §6.1 / `[code]` |
| learning rate | 1.0, decayed to 0.1 at step 336,000 and kept there | `[paper]` §6.1 / `[code]` |
| batch size | 32 | `[paper]` §6.1 |
| warmup / grad clip / weight decay | none / global norm 5 / 0 | `[ours]` / `[code]` / `[ours]` |
| embedding L2, dropout | 0 / 0 | `[ours]` the paper applies MBA only on Alibaba |
| embedding dim | 64 item + 64 category | `[code]` |
| activation | PReLU (attention MLP [80, 40], DNN [80, 40]) | `[paper]` Table 3 fn. a / `[code]` |
| attention | no softmax (weights keep their intensity) | `[paper]` §4.3 |
| loss / metric | BCE / user-weighted AUC (gAUC) | `[paper]` Eq. 2 / Eq. 10 |
| epochs | 50, no early stopping | `[code]` |
| checkpoint selection | best test gAUC (the released split has no validation set) | `[code]` |
| seed | 1234 | `[code]` |

Paper target (Table 3, DIN with PReLU): **gAUC 0.8818** (DIN + Dice: 0.8871).

## Curve

The learning rate drops from 1.0 to 0.1 at epoch 5 (step 336,000) and then stays
at 0.1.  gAUC peaks at epoch 7 and declines afterwards — the run continues to
epoch 50 because that is what the authors' code does, and the best checkpoint is
saved at every improvement.

| Epoch | LR | Train loss | gAUC | AUC | Epoch time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.5448 | 0.8245 | 0.8255 | 1019 |
| 2 | 1.0 | 0.4994 | 0.8344 | 0.8354 | 1046 |
| 3 | 1.0 | 0.4741 | 0.8400 | 0.8398 | 1032 |
| 4 | 1.0 | 0.4529 | 0.8534 | 0.8510 | 1024 |
| 5 | 1.0 | 0.4033 | 0.8543 | 0.8550 | 1023 |
| 6 | 0.1 | 0.3799 | 0.8528 | 0.8543 | 1023 |
| **7 (best)** | **0.1** | **0.3673** | **0.8555** | **0.8564** | 1012 |
| 8 | 0.1 | 0.3548 | 0.8518 | 0.8527 | 1306 |
| 9 | 0.1 | 0.3420 | 0.8531 | 0.8534 | 1029 |
| 10 | 0.1 | 0.3296 | 0.8480 | 0.8491 | 1029 |
| 11 | 0.1 | 0.3176 | 0.8442 | 0.8445 | 1446 |
| 12 | 0.1 | 0.3059 | 0.8412 | 0.8413 | 1020 |
| 13 | 0.1 | 0.2949 | 0.8375 | 0.8371 | 1024 |
| 14 | 0.1 | 0.2839 | 0.8328 | 0.8319 | 1047 |
| 15 | 0.1 | 0.2734 | 0.8352 | 0.8354 | 1431 |
| 16 | 0.1 | 0.2631 | 0.8339 | 0.8337 | 1523 |

Epoch times above 1,050 s are periods when another job shared the host.

## Results

| Metric | Best epoch (7) | Paper |
| --- | --- | --- |
| gAUC | **0.8555** | 0.8818 |
| AUC | 0.8564 | — |

Final numbers will replace this table when the 50-epoch run ends; the current
best is already saved.

## Notes

* **First attempt aborted (LR schedule bug).**  The first version of the
  scheduler multiplied the rate by 0.1 every 336,000 steps, so it reached 1e-9
  by epoch 39 and stopped learning (best gAUC 0.8555 at epoch 7).  The authors'
  code assigns `lr = 0.1` once, so the rate stays at 0.1; `min_lr: 0.1` now
  reproduces that.  Up to epoch ~10 both runs are identical (same seed), which is
  why the best number is unchanged — the model overfits after epoch 7 either way.
* **Gap to the paper.**  The same data and the BARS/FuxiCTR configuration
  (`din.yaml`) reach gAUC 0.8835, so the 0.026 gap comes from the paper's own
  choices rather than from the dataset: PReLU instead of the code's sigmoid,
  no softmax in the activation unit, SGD 1.0 with batch 32 instead of Adam 5e-4
  with batch 1024, and MLPs of [80, 40] instead of [1024, 512, 256].
* **Model selection uses the test split** because the released split has no
  validation set — the authors' `_eval` does the same.  The number is therefore
  best-over-epochs test gAUC, not an untouched held-out score.
* 50 epochs at ~1,030 s each is ~14.5 h; the run started 10:06 and is expected
  to finish around 00:45–02:00 on 2026-10-30.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best gAUC epoch weights (rewritten on improvement) |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | written at the end: summary + full history |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=din_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
