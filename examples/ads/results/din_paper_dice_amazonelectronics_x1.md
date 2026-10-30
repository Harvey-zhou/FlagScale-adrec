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
| pending | | | | |

## Results

Pending — the final table will replace this section when the 50-epoch run ends.

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
