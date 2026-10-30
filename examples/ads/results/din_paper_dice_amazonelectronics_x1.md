# DIN with Dice — Amazon Electronics_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **running** (started 2026-10-30 09:51 CST) |
| Config | `examples/ads/conf/train/din_paper_dice.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | `6b52a10` (includes `e463e1e`, PReLU/Dice slope init 0) |
| Device | Kunlunxin P800, card 6, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 4,152,226 (same shape as `din_paper`, PReLU replaced by Dice) |
| Output | `/data2/jiangxuhui/logs/ads-paper-dice/ckpts/din_paper_dice-cuda/checkpoint/` |
| Command | `launch-ads-paper.sh 6 din_paper_dice /data2/jiangxuhui/logs/ads-paper-dice` |

## Settings

Identical to the [DIN paper run](din_paper_amazonelectronics_x1.md) except for
the activation, which is the paper's "DIN with Dice" row (Table 3, 0.8871):

| Setting | Value | Source |
| --- | --- | --- |
| activation | Dice in the activation unit and the final MLP | `[paper]` Table 3, §5.2 |
| Dice epsilon | 1e-8 | `[paper]` §5.2 ("set to be 1e-8 in our practice") |
| Dice slope init | 0 | `[paper]` Eq. (9) form / `[code]` (its Dice alpha starts at 0) |
| everything else | SGD 1.0 → 0.1 at step 336,000, batch 32, no warmup/weight decay/dropout, global-norm clip 5, embedding 64+64, attention MLP [80, 40], DNN [80, 40], no softmax, 50 epochs, seed 1234 | see the DIN run above |

Paper target (Table 3): **gAUC 0.8871** (paper's own TF run: 0.8711).

## Curve

One epoch ≈ 15 min; the first row arrives ~15 minutes after the data load.

| Epoch | LR | Train loss | gAUC | AUC |
| --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.5382 | 0.8277 | 0.8284 |
| 2 | 1.0 | 0.4911 | 0.8343 | 0.8358 |
| 3 | 1.0 | 0.4643 | 0.8477 | 0.8493 |
| 4 | 1.0 | 0.4426 | 0.8540 | 0.8556 |
| **5 (best so far)** | **1.0** | **0.3927** | **0.8575** | **0.8587** |
| 6 | 0.1 | 0.3676 | 0.8539 | 0.8546 |
| 8 | 0.1 | 0.3393 | 0.8543 | 0.8549 |
| 10 | 0.1 | 0.3096 | 0.8470 | 0.8461 |
| 12 | 0.1 | 0.2800 | 0.8390 | 0.8376 |
| 14 | 0.1 | 0.2522 | 0.8312 | 0.8301 |
| 16 | 0.1 | 0.2264 | 0.8277 | 0.8266 |
| 18 | 0.1 | 0.2033 | 0.8226 | 0.8214 |
| 20 | 0.1 | 0.1822 | 0.8154 | 0.8142 |
| 22 | 0.1 | 0.1636 | 0.8098 | 0.8081 |

As in the PReLU run, the model peaks just before the learning-rate drop and then
overfits while the training loss keeps falling.  The run continues to epoch 50;
the best checkpoint is already saved.

## Results

Interim (best epoch 5): gAUC **0.8575** / AUC 0.8587, versus 0.8555 / 0.8564 for
the PReLU run and the paper's 0.8871.  Dice is worth about +0.002 here, far
from the paper's +0.0053 over DIN — but the comparison is confounded: the PReLU
run started the activation slope at 0.25 (pre-`e463e1e`) and this one starts at
0, and neither matches the structure ablations above (the released structure is
worth ~0.03).  The final table will replace this section at epoch 50.

## Notes

* The PReLU DIN run trained before commit `e463e1e` (PReLU slope default 0.25);
  this run starts from 0 as the paper's Eq. (9) form suggests.  The two runs are
  therefore not a perfectly controlled PReLU-vs-Dice comparison: a matching
  PReLU baseline would need a rerun with `init=0.0`.
* Same data, metric and test-set model selection as the DIN run: the released
  split has no validation set.

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
  action=test train=din_paper_dice train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
