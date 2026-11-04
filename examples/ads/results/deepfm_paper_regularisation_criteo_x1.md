# DeepFM — Criteo_x1, why the paper protocol trails BARS (regularisation decomposition)

| Field | Value |
| --- | --- |
| Status | **finished** (three single-variable arms) |
| Arms | paper config + BatchNorm; + embedding L2 1e-5; + all three BARS regularisers |
| Started / finished | 2026-11-03 22:52 → 23:24 (BN), → 2026-11-04 00:57 (L2), → 02:19 (all three) |
| Config | `examples/ads/conf/train/deepfm_paper.yaml` + one override |
| Base results | [deepfm_paper_criteo_x1.md](deepfm_paper_criteo_x1.md), [lr ablation](deepfm_paper_lr_ablation_criteo_x1.md) |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, cards 5 (BN) and 6 (L2), vendor torch 2.9.0+cu128, fp32 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/deepfm-paper-bn/`, `.../deepfm-paper-l2/` |

## Why these arms

The lr sweep left 0.0057 AUC to the same-host BARS config and showed that no
step size stops the early overfitting (every arm peaks in 1–3 epochs).  The
remaining paper-silent differences against BARS are **BatchNorm** and
**embedding L2 1e-5** — BARS adds both, the paper describes neither.  Each is
tested alone, on top of the best lr (3e-4), against a plain 3e-4 baseline.

These arms are deliberately **not** paper-protocol runs: they say what the gap
is made of, and the paper-protocol headline stays at 0.8040.

## Curve (validation AUC)

| Epoch | plain (3e-4) | + BatchNorm | + embedding L2 | + all three |
| --- | --- | --- | --- |
| 1 | 0.80371 | 0.80167 | 0.80231 | 0.80373 |
| 2 | **0.80513** | **0.80415** | 0.80529 | 0.80726 |
| 3 | 0.80322 | 0.80229 | 0.80678 | 0.80882 |
| 4 | 0.79998 | 0.79868 | 0.80769 | 0.80944 |
| 5 | 0.79683 | 0.79500 | 0.80827 | 0.81001 |
| 7 | 0.79209 | 0.79042 | 0.80887 | 0.81049 |
| 8 | — | — | 0.80893 | 0.81053 |
| **9** | — | — | 0.80899 | **0.81059 (best)** |
| 10 | — | — | 0.80916 | 0.81052 |
| 11 | — | — | 0.80931 | 0.81046 |
| 13 | — | — | 0.80945 | 0.81049 |
| 14 (last) | — | — | 0.80947 | 0.81028 |
| 19 | — | — | 0.80972 | — |
| **21** | — | — | **0.80980 (best)** | — |
| 26 (last) | — | — | 0.80969 | — |

BatchNorm changes nothing structural: it peaks at epoch 2 like the plain arm and
then declines the same way (it is a stability device here, not an accuracy one —
the same conclusion the DIN ablation reached).  Embedding L2 changes the shape
of the run completely: it keeps improving for 21 epochs.  With all three
regularisers the curve tracks the BARS run: it peaks at epoch 9 — the same epoch
BARS peaks at — and then flattens.

## Results

| Configuration | test AUC | test logloss | best epoch | Δ AUC vs BARS |
| --- | --- | --- | --- | --- |
| Paper protocol (lr 1e-3) | 0.8040 | 0.4477 | 1 | −0.0072 |
| + lr 3e-4 | 0.8055 | 0.4466 | 2 | −0.0057 |
| + BatchNorm (lr 3e-4) | 0.8045 | 0.4469 | 2 | −0.0067 |
| **+ embedding L2 1e-5 (lr 3e-4)** | **0.8102** | **0.4428** | 21 | **−0.0010** |
| **+ L2 + BatchNorm + dropout 0.1** | **0.8111** | **0.4406** | 9 | **−0.0001** |
| BARS `deepfm`, this host | 0.8112 | 0.4406 | 9 | — |
| Paper (unavailable 90/10 split) | 0.8007 | 0.45083 | — | — |

**The 0.0072 AUC gap is the regularisation, and nothing else.** Adding the
embedding L2 term alone brings the paper's architecture to 0.8102; adding the
other two BARS regularisers (BatchNorm, dropout 0.1 instead of 0.5) on top lands
at **0.81108 / 0.440621 against BARS's 0.811187 / 0.440624** — a 0.0001 AUC
difference.  The paper's 400-400-400 / embedding-10 architecture is therefore
not what separates the two runs; the missing regularisers are.  This also means
the paper protocol's 0.8040 is a *correct* rendering of the paper's recipe on
this data, not an implementation shortfall.

The best epoch (9) of the full-regularisation arm is also identical to BARS's
best epoch (9), which is what one expects when the two configurations have
become numerically the same.

The paper's own number (0.8007) sits below all of these because it comes from a
different, undistributed split.

## Cost

| Arm | Epochs | Epoch time | Mean samples/s |
| --- | --- | --- | --- |
| + BatchNorm | 7 | ~265 s | 130,303 |
| + embedding L2 | 26 | ~287 s | 117,168 |
| + L2 + BatchNorm + dropout 0.1 | 14 | ~350 s | 98,317 |
| plain 3e-4 | 7 | ~200 s | 163,215 |

The L2 regulariser is applied to every embedding parameter, so it costs ~28 % of
training throughput on this stack (163 k → 117 k samples/s) — the price of the
0.0047 AUC it buys.  BatchNorm adds a further ~16 % (117 k → 98 k).

## Artifacts

| Path | Contents |
| --- | --- |
| `.../deepfm-paper-bn/ads/ckpts/deepfm_paper-cuda/checkpoint/` | metrics.json, history.json, model.pt |
| `.../deepfm-paper-l2/ads/ckpts/deepfm_paper-cuda/checkpoint/` | same |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=deepfm_paper train.optimizer.lr=3e-4 \
  train.optimizer.embedding_regularizer=1e-5 \
  train.hardware.device_type=cuda experiment.runner.nproc_per_node=1
```
