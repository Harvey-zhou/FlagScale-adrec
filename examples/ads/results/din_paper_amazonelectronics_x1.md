# DIN — Amazon Electronics_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (all 50 epochs) |
| Started / finished | 2026-10-29 10:06:07 → 23:52 (CST) |
| Wall clock | 13 h 46 m (training 49,547 s) |
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
| 17 | 0.1 | 0.2534 | 0.8277 | 0.8269 | 811 |
| 18 | 0.1 | 0.2436 | 0.8266 | 0.8256 | 787 |
| 19 | 0.1 | 0.2342 | 0.8267 | 0.8254 | 792 |
| 20 | 0.1 | 0.2251 | 0.8242 | 0.8221 | 873 |
| 21 | 0.1 | 0.2166 | 0.8225 | 0.8219 | 841 |
| 22 | 0.1 | 0.2087 | 0.8216 | 0.8207 | 898 |
| 23 | 0.1 | 0.2005 | 0.8148 | 0.8139 | 905 |
| 24 | 0.1 | 0.1932 | 0.8156 | 0.8139 | 897 |
| 25 | 0.1 | 0.1857 | 0.8150 | 0.8144 | 821 |
| 26 | 0.1 | 0.1792 | 0.8118 | 0.8103 | 809 |
| 27 | 0.1 | 0.1730 | 0.8062 | 0.8050 | 820 |
| 28 | 0.1 | 0.1666 | 0.8096 | 0.8076 | 828 |
| 29 | 0.1 | 0.1602 | 0.8149 | 0.8149 | 847 |
| 30 | 0.1 | 0.1554 | 0.8096 | 0.8088 | 869 |
| 35 | 0.1 | 0.1310 | 0.8019 | 0.8003 | 902 |
| 40 | 0.1 | 0.1130 | 0.8039 | 0.8032 | 880 |
| 45 | 0.1 | 0.0988 | 0.7971 | 0.7968 | 910 |
| 50 (last) | 0.1 | 0.0881 | 0.7968 | 0.7953 | 910 |

Epoch times of 1,300–1,500 s around epochs 8–16 were periods when benchmark
jobs shared the host; the steady state is ~800–900 s per epoch.

## Results

| Metric | Best epoch (7) | Final epoch (50) | Paper |
| --- | --- | --- | --- |
| gAUC | **0.8555** | 0.7968 | 0.8818 |
| AUC | 0.8564 | 0.7953 | — |
| logloss | 0.4828 | — | — |

The model overfits from epoch 8 on (train loss 0.367 → 0.088 while test gAUC
falls 0.8555 → 0.7968).  The reported number is the best checkpoint, which is
what the authors' code restores before its final test pass.

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
* **PReLU initial slope.**  This run was launched before commit `e463e1e`, which
  changed `nn.PReLU(width)` (PyTorch default slope 0.25) to `init=0.0`.  The
  paper does not state the initial α, and the released code has no PReLU at all
  (both of its MLPs use sigmoid), so neither value contradicts the paper — but
  the protocol document now describes 0.0.  Everything else in this table is
  unaffected; an exact match to the current code would need a rerun.
* 50 epochs took 13 h 46 m (median 905 s/epoch; 2,882 samples/s).  The
  1,300–1,500 s epochs were periods when benchmark jobs shared the host.

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
