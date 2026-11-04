# DIN with Dice — Amazon Electronics_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (50 epochs; diverged to NaN at epoch 38, best checkpoint epoch 5) |
| Started / finished | 2026-11-04 00:57:20 → 09:22 (CST) — rerun; the first run was 2026-10-30 09:51 → 10-31 03:11 |
| Wall clock | 8 h 24 m (first run: 17 h 21 m) |
| Config | `examples/ads/conf/train/din_paper_dice.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | `dc5cb07` (first run: `6b52a10`) |
| Device | Kunlunxin P800, card 6, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 4,152,226 (same shape as `din_paper`, PReLU replaced by Dice) |
| Output | `/data2/jiangxuhui/logs/adrec-runs/din-paper-dice-rerun/ads/ckpts/din_paper_dice-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh ads din_paper_dice 6` |

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

One epoch ≈ 10 min on an idle card (≈ 605 s, 4,398 samples/s); the first run
cost ~15 min per epoch because the card was shared.

| Epoch | LR | Train loss | gAUC | AUC |
| --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.5382 | 0.8277 | 0.8284 |
| 2 | 1.0 | 0.4911 | 0.8343 | 0.8358 |
| 3 | 1.0 | 0.4643 | 0.8477 | 0.8493 |
| 4 | 1.0 | 0.4426 | 0.8540 | 0.8556 |
| **5 (best)** | **1.0** | **0.3927** | **0.8575** | **0.8587** |
| 6 | 0.1 | 0.3676 | 0.8539 | 0.8546 |
| 8 | 0.1 | 0.3393 | 0.8543 | 0.8549 |
| 10 | 0.1 | 0.3096 | 0.8470 | 0.8461 |
| 12 | 0.1 | 0.2800 | 0.8390 | 0.8376 |
| 14 | 0.1 | 0.2522 | 0.8312 | 0.8301 |
| 16 | 0.1 | 0.2264 | 0.8277 | 0.8266 |
| 18 | 0.1 | 0.2033 | 0.8226 | 0.8214 |
| 20 | 0.1 | 0.1822 | 0.8154 | 0.8142 |
| 22 | 0.1 | 0.1636 | 0.8098 | 0.8081 |
| 30 | 0.1 | 0.1120 | 0.7994 | 0.7983 |
| 37 | 0.1 | 0.0860 | 0.8053 | 0.8040 |
| 38 | 0.1 | **NaN** | 0.0 | 0.5 |
| 50 | 0.1 | NaN | 0.0 | 0.5 |

As in the PReLU run, the model peaks just before the learning-rate drop and then
overfits while the training loss keeps falling.  At **epoch 38 the loss became
NaN** and every later evaluation reports AUC 0.5 / gAUC 0.0; the best checkpoint
(epoch 5) was already saved and is unaffected.  The divergence is a real
property of this configuration on this stack — SGD at a constant 0.1 with Dice
activations, even with the global-norm clip of 5 that the author code uses —
and it is recorded rather than hidden.

## Results

Final (best epoch 5): gAUC **0.8575** / AUC 0.8587 / logloss 0.4816.  The rerun
reproduced the first run exactly through epoch 8 (0.8277 / 0.8343 / 0.8477 /
0.8540 / 0.8575 / 0.8539 / 0.8548 / 0.8543) and **NaN'd at the same epoch 38**,
so both the result and the divergence are deterministic properties of this
configuration, not run-to-run noise.

With the PReLU run now rerun at the same initialisation, the comparison is
controlled for the first time:

| Activation | gAUC | AUC | best epoch |
| --- | --- | --- | --- |
| PReLU (init 0), current code | **0.8645** | 0.8649 | 11 |
| Dice (alpha 0), current code | 0.8575 | 0.8587 | 5 |
| Paper Table 3 | 0.8871 | — | — |
| Authors' README (their TF run) | 0.8711 | — | — |

**Dice is 0.0070 *worse* than PReLU on this data**, while the paper reports
+0.0053 the other way and the authors' own README +0.0013.  Same data, seed,
protocol and initialisation, so this is a property of the Dice implementation
on this stack (BN gate at batch 32, SGD 1.0, no dropout/regulariser) rather
than of the two runs being incomparable.  The released-structure Dice run
below reaches 0.8658 at epoch 2 and then NaN's at epoch 3, which is the same
instability.

## Notes

* The PReLU-vs-Dice comparison is now controlled: the PReLU arm was rerun on
  the same code, data, seed and protocol and reaches 0.8645 (see the
  [DIN run](din_paper_amazonelectronics_x1.md)).
* The NaN is reproducible: two runs with different code revisions (init-0.25
  and init-0 PReLU elsewhere in the model) both die at **epoch 38**, with the
  train loss falling normally to 0.086 before the first non-finite batch.  The
  best checkpoint (epoch 5) predates it, so the recorded number is unaffected.
  The crash is consistent with `alpha`/BN statistics diverging under SGD at a
  constant 0.1 with batch 32 and no dropout; a future fix would clamp the Dice
  gate or train it at a lower rate, but that is no longer the paper's protocol.
* Same data, metric and test-set model selection as the DIN run: the released
  split has no validation set.

## Released structure (Dice)

The paper's 0.8871 row was produced by `model_dice.py`, which keeps **sigmoid**
in the attention MLP, uses Dice only in the DNN, still normalises the attention
with softmax after `sqrt(d)`, and applies `hist_bn`/`hist_fcn`, `[u, i, u*i]`
and BatchNorm on that input.  Config `din_dice_author.yaml` keeps the same
protocol and switches those pieces on:

| Epoch | gAUC | AUC |
| --- | --- | --- |
| 1 | 0.8597 | 0.8614 |
| **2 (best)** | **0.8658** | **0.8681** |
| 3 | NaN | 0.500 |

The run diverged to NaN at epoch 3 (constant SGD 0.1 with Dice), so the best
checkpoint is epoch 2.  0.8658 is +0.0083 over the paper-structure Dice run
(0.8575) and within 0.004 of the PReLU released-structure run (0.8696 at epoch
3) — i.e. Dice and PReLU are indistinguishable in this structure, and the
paper's 0.8871 remains ~0.02 out of reach, the same gap the paper structure
shows on this data/implementation.

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
