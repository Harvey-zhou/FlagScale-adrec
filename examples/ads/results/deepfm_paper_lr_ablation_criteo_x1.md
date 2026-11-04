# DeepFM — Criteo_x1, learning-rate ablation (paper protocol)

| Field | Value |
| --- | --- |
| Status | **finished** (three arms; early stopping decided each) |
| Arms | `lr=1e-3` (paper protocol), `lr=3e-4`, `lr=1e-4` |
| Started / finished | 2026-11-03 22:21 → 22:46 (3e-4), → 22:50 (1e-4) |
| Config | `examples/ads/conf/train/deepfm_paper.yaml` + `train.optimizer.lr=<arm>` |
| Base result | [deepfm_paper_criteo_x1.md](deepfm_paper_criteo_x1.md) |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, cards 5 (3e-4) and 6 (1e-4), vendor torch 2.9.0+cu128, fp32 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/deepfm-paper-lr3e-4/`, `.../deepfm-paper-lr1e-4/` |

## Why this ablation

The paper-protocol run (lr 1e-3) peaks at **epoch 1** and declines every epoch
after that, while the BARS configuration trains 11 epochs and keeps improving.
The base document named the learning rate as the cheapest explanation: at 1e-3
the model may simply be stepping too far on this 33 M-row training set.  This
sweep tests that, as the base document proposed.

Everything except `optimizer.lr` is identical: 400-400-400 DNN, ReLU, dropout
0.5, no BatchNorm, no embedding L2, Adam, batch 4096, `early_stop_patience` 5.

## Curve (validation AUC)

| Epoch | lr 1e-3 | lr 3e-4 | lr 1e-4 |
| --- | --- | --- | --- |
| 1 | **0.80360 (best)** | 0.80371 | 0.80064 |
| 2 | 0.80160 | **0.80513 (best)** | 0.80329 |
| 3 | 0.79780 | 0.80322 | **0.80333 (best)** |
| 4 | 0.79300 | 0.79998 | 0.80202 |
| 5 | 0.78940 | 0.79683 | 0.79944 |
| 6 | 0.78560 | 0.79332 | 0.79620 |
| 7 | — | 0.79209 | 0.79495 |
| 8 | — | — | 0.79176 |

Lowering the rate delays the peak (epoch 1 → 2 → 3) and lowers the train-loss
floor it reaches first, but **every arm still turns down after 1–3 epochs**: the
configuration is overfitting the training set regardless of the step size, not
oscillating at a too-large step.

## Results

| Metric | lr 1e-3 | lr 3e-4 | lr 1e-4 | BARS `deepfm`, this host |
| --- | --- | --- | --- | --- |
| test AUC | 0.8040 | **0.805514** | 0.803726 | 0.811187 |
| test logloss | 0.4477 | **0.446628** | 0.447895 | 0.440624 |
| best epoch | 1 | 2 | 3 | 9 |
| Δ AUC vs BARS | −0.0072 | −0.0057 | −0.0075 | — |

The sweep is monotone with a maximum at 3e-4 and buys **0.0015 AUC**
(21 % of the same-host BARS gap); moving to 1e-4 gives it back.  The learning
rate is therefore *a* factor but not the explanation.

## What the sweep rules out

Combined with the base run, the picture is:

* the paper configuration has no regulariser beyond dropout 0.5 and no
  BatchNorm, while BARS adds BatchNorm **and** embedding L2 1e-5 and uses
  dropout 0.1 — and BARS is the run that keeps improving for 11 epochs;
* the paper's own 0.8007 comes from a 90/10 split of the 45 M-row Kaggle file
  that is not distributed, so the paper-protocol run cannot be calibrated
  against it;
* every lr arm peaks early and degrades, which is the signature of a missing
  regulariser rather than a step-size problem.

The next cheap check is therefore **paper config + embedding L2 1e-5** (one
variable, ~25 min per arm), not another lr point.  That is a deliberate
departure from the paper protocol, so it is listed here as a follow-up rather
than folded into the paper result.

## Cost

| Arm | Epochs run | Epoch time | Mean samples/s |
| --- | --- | --- | --- |
| 1e-3 | 6 | ~300 s | ~110,000 (card shared) |
| 3e-4 | 7 | ~200 s | 163,215 |
| 1e-4 | 8 | ~205 s | 158,123 |

8,058 steps per epoch at batch 4096.

## Artifacts

| Path | Contents |
| --- | --- |
| `.../deepfm-paper-lr3e-4/ads/ckpts/deepfm_paper-cuda/checkpoint/` | metrics.json, history.json, model.pt |
| `.../deepfm-paper-lr1e-4/ads/ckpts/deepfm_paper-cuda/checkpoint/` | same |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=deepfm_paper train.optimizer.lr=3e-4 \
  train.hardware.device_type=cuda experiment.runner.nproc_per_node=1
```
