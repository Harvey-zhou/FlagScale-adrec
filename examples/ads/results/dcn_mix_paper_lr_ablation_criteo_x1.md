# DCN-Mix — Criteo_x1, learning-rate ablation

| Field | Value |
| --- | --- |
| Status | **finished** (three arms, 3 epochs each) |
| Arms | `lr=1e-4`, `lr=5e-4` (paper protocol), `lr=1e-3` |
| Started / finished | 1e-3: 2026-11-03 19:29 → 21:04; 1e-4: 19:33 → 20:59 |
| Config | `examples/ads/conf/train/dcn_mix_paper.yaml` + `train.optimizer.lr=<arm>` |
| Base result | [dcn_mix_paper_criteo_x1.md](dcn_mix_paper_criteo_x1.md) |
| Code commit | `dc5cb07` |
| Device | Kunlunxin P800, cards 4 (1e-3) and 3 (1e-4), vendor torch 2.9.0+cu128, fp32 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/dcnmix-lr1e-3/`, `.../dcnmix-lr1e-4/` |

## Why this ablation

The paper protocol run (lr 5e-4) landed 0.0035 AUC below the BARS DCN-V2
reference on the same split (0.8104 vs 0.8139, the BARS config re-run on this
host — see [bars_baseline_criteo_x1.md](bars_baseline_criteo_x1.md)).  The paper
only narrows Adam to 1e-4…5e-4 and does not publish the split, so the cheapest
unexplained variable is the learning rate: the paper protocol took the **top** of
the range while BARS uses **1e-3**.  This tests both directions around the
original run.

Everything except `optimizer.lr` is identical across the three arms: batch 512,
3 epochs (193,380 steps), seed 2021, embedding 39, 3 cross layers, 4 experts,
low rank 64, hard_tanh activation, constant gate, He init, DNN [512, 512],
EMA 0.9999, gradient clip 10, embedding L2 1e-5.

## Curve (validation, every epoch)

| Epoch | lr 1e-4 AUC | lr 5e-4 AUC | lr 1e-3 AUC | lr 1e-4 logloss | lr 5e-4 logloss | lr 1e-3 logloss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.80657 | 0.8071 | 0.80544 | 0.44488 | 0.4445 | 0.44602 |
| 2 | 0.80995 | 0.8094 | 0.80696 | 0.44168 | 0.4423 | 0.44445 |
| **3 (best)** | **0.81116** | **0.8102** | **0.80720** | **0.44053** | **0.4415** | **0.44420** |

Train loss at epoch 3: 0.44733 (1e-4), 0.4559 (5e-4), 0.46763 (1e-3).  The
1e-3 arm's train loss is *flat between epochs 2 and 3* (0.46590 → 0.46763) —
it is already bouncing at the bottom rather than converging.

## Results

| Metric | lr 1e-4 | lr 5e-4 (paper protocol) | lr 1e-3 | BARS `dcn_mix`, this host | Paper (other split) |
| --- | --- | --- | --- | --- | --- |
| test AUC | **0.811514** | 0.810421 | 0.807456 | 0.813934 | 0.8112 |
| test logloss | **0.440073** | 0.441089 | 0.443815 | 0.437975 | 0.4408 |
| Δ AUC vs BARS | −0.0024 | −0.0035 | −0.0065 | — | — |

Ordering is monotone in the learning rate: **1e-4 > 5e-4 > 1e-3**.  Lowering the
rate by 5× recovered 0.0011 AUC (31 % of the same-host BARS gap) and 0.0010
logloss, while raising it to BARS's 1e-3 cost 0.0030 AUC.  The hypothesis the
ablation was built on — that BARS's faster rate explains part of the gap — is
therefore **falsified**; the direction is the opposite.

At 1e-4 all three epochs still improve on validation (0.80657 → 0.80995 →
0.81116) and the train loss is still falling, so this arm is also truncated by
the 3-epoch budget, more so than the 5e-4 arm.  The 193 k-step budget is the
paper's; the remaining gap cannot be separated from the step budget by this
experiment.

## Reading of the BARS gap

Still not paper-specified, since BARS does not follow the paper's best setting:

| | Paper protocol (this run) | BARS `DCNv2_criteo_x1` |
| --- | --- | --- |
| embedding size | 39 | 10 |
| low rank | 64 / expert | 32 |
| DNN | [512, 512] | [500, 500, 500] |
| dropout | 0 | 0.1 |
| Adam lr | 1e-4 … 5e-4 | 1e-3 |
| batch | 512 | 4096 |
| mixture | on | off (published as full-rank DCN-V2) |

The lr arm moves in the direction of BARS only in the sense that the *lower*
rate is better on this 3-epoch budget; the rest of the BARS recipe (embedding 10,
low rank 32, 500³ DNN, dropout 0.1, batch 4096) is still untested here, and the
mixture itself costs ~0.0003 AUC on the paper's own split.

## Cost

| Arm | Epoch 1 / 2 / 3 (s) | Samples/s |
| --- | --- | --- |
| 1e-4 | 1,695 / 1,705 / 1,680 | 19,468 / 19,355 / 19,640 |
| 5e-4 | 3,443 / 3,509 / 3,383 | ~9,580 (card shared) |
| 1e-3 | 1,912 / 1,883 / 1,832 | 17,261 / 17,527 / 18,015 |

33.0 M rows per epoch at batch 512 = 64,460 steps.  The 5e-4 arm ran while the
card was shared with other users; the 1e-4 and 1e-3 arms overlapped each other
on two cards, which is why 1e-3 is ~10 % slower per epoch than 1e-4 despite the
same model.  Steps and epoch counts are identical across arms, so the quality
comparison is unaffected.

## Artifacts

| Path | Contents |
| --- | --- |
| `.../dcnmix-lr1e-4/ads/ckpts/dcn_mix_paper-cuda/checkpoint/` | metrics.json, history.json, model.pt (339 MB) |
| `.../dcnmix-lr1e-3/ads/ckpts/dcn_mix_paper-cuda/checkpoint/` | same |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=dcn_mix_paper train.optimizer.lr=1e-4 \
  train.hardware.device_type=cuda experiment.runner.nproc_per_node=1
```
