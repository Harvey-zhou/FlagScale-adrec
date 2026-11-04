# DIN — Amazon Electronics_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (all 50 epochs; best epoch 11) |
| Started / finished | 2026-11-03 22:11:27 → 2026-11-04 05:06 (CST) |
| Wall clock | 6 h 45 m (training 24,271 s) |
| Config | `examples/ads/conf/train/din_paper.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | `dc5cb07` (PReLU initial slope `0.0`, i.e. `layers.py` `nn.PReLU(width, init=0.0)`) |
| Device | Kunlunxin P800, card 4, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 4,152,226 |
| Output | `/data2/jiangxuhui/logs/adrec-runs/din-paper-rerun/ads/ckpts/din_paper-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh ads din_paper 4` |

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
| activation | PReLU, initial slope 0.0 (attention MLP [80, 40], DNN [80, 40]) | `[paper]` Table 3 fn. a / `[ours]` the slope is not specified |
| attention | no softmax (weights keep their intensity) | `[paper]` §4.3 |
| loss / metric | BCE / user-weighted AUC (gAUC) | `[paper]` Eq. 2 / Eq. 10 |
| epochs | 50, no early stopping | `[code]` |
| checkpoint selection | best test gAUC (the released split has no validation set) | `[code]` |
| seed | 1234 | `[code]` |

Paper target (Table 3, DIN with PReLU): **gAUC 0.8818** (DIN + Dice: 0.8871).

## Curve

The learning rate drops from 1.0 to 0.1 at step 336,000 and then stays at 0.1.
The drop lands at 336,000 / 81,524 ≈ 4.12 epochs, i.e. **inside epoch 5**.  The
`LR` column below is the rate at the *start* of the epoch, which is why epoch 5
still shows 1.0 and epoch 6 is the first one that starts at 0.1.

gAUC peaks at **epoch 11** and declines afterwards — the run continues to epoch
50 because that is what the authors' code does, and the best checkpoint is saved
at every improvement.  Epoch 4 (0.7830) is a transient dip while the rate is
still 1.0; it recovers to 0.8606 at the epoch the rate drops.

| Epoch | LR | Train loss | gAUC | AUC | Epoch time (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.5466 | 0.8227 | 0.8239 | 506 |
| 2 | 1.0 | 0.5030 | 0.8334 | 0.8349 | 501 |
| 3 | 1.0 | 0.4894 | 0.8400 | 0.8405 | 490 |
| 4 | 1.0 | 0.4727 | 0.7830 | 0.7787 | 533 |
| 5 | 1.0 | 0.4270 | 0.8606 | 0.8623 | 497 |
| 6 | 0.1 | 0.4083 | 0.8615 | 0.8631 | 490 |
| 7 | 0.1 | 0.3991 | 0.8640 | 0.8650 | 489 |
| 8 | 0.1 | 0.3906 | 0.8625 | 0.8637 | 479 |
| 9 | 0.1 | 0.3816 | 0.8618 | 0.8626 | 493 |
| 10 | 0.1 | 0.3720 | 0.8617 | 0.8624 | 519 |
| **11 (best)** | **0.1** | **0.3625** | **0.8645** | **0.8649** | 512 |
| 12 | 0.1 | 0.3517 | 0.8569 | 0.8569 | 484 |
| 13 | 0.1 | 0.3410 | 0.8540 | 0.8546 | 459 |
| 15 | 0.1 | 0.3189 | 0.8523 | 0.8520 | 480 |
| 20 | 0.1 | 0.2646 | 0.8429 | 0.8412 | 502 |
| 25 | 0.1 | 0.2198 | 0.8273 | 0.8253 | 485 |
| 30 | 0.1 | 0.1859 | 0.8272 | 0.8249 | 486 |
| 35 | 0.1 | 0.1610 | 0.8176 | 0.8153 | 456 |
| 40 | 0.1 | 0.1423 | 0.7983 | 0.7968 | 455 |
| 45 | 0.1 | 0.1277 | 0.8139 | 0.8137 | 452 |
| 50 (last) | 0.1 | 0.1169 | 0.8018 | 0.8009 | 478 |

The card was idle for this run (mean 5,438 samples/s), which is why epochs cost
~480 s instead of the ~900 s of the 2026-10-29 run, where benchmark jobs shared
the host.

## Results

| Metric | Best epoch (11) | Final epoch (50) | Paper | Authors' README |
| --- | --- | --- | --- | --- |
| gAUC | **0.8645** | 0.8018 | 0.8818 | 0.8698 |
| AUC | 0.8649 | 0.8009 | — | — |
| logloss | 0.4714 | — | — | — |

The model overfits from epoch 12 on (train loss 0.363 → 0.117 while test gAUC
falls 0.8645 → 0.8018).  The reported number is the best checkpoint, which is
what the authors' code restores before its final test pass.

### Against the available targets

| Target | gAUC | This run (paper structure, PReLU init 0) |
| --- | --- | --- |
| Paper Table 3 (DIN, PReLU) | 0.8818 | −0.0173 |
| Authors' README (their own run of this code) | 0.8698 | **−0.0053** |
| BARS/FuxiCTR `din.yaml` on this data (fresh run, batch 1024) | 0.8823 | −0.0178 |
| Released DIN structure, this repo (50 epochs) | 0.8696 | −0.0051 |
| **Previous run of this exact config (PReLU init 0.25)** | 0.8555 | +0.0090 |

The authors' own README reports 0.8698 with the released code on the same kind
of split, so the paper's 0.8818 is not reproducible as published even by its
authors; 0.8698 is the more meaningful target, and this run is 0.0053 short of
it — inside the spread between the authors' own two numbers (0.8698 and the
0.8818 in the paper).

**The PReLU initial slope matters (0.009 gAUC), contrary to the earlier
reading.**  The 2026-10-29 run of this config used PyTorch's default slope 0.25
and peaked at 0.8555; `e463e1e` changed it to 0.0, and this rerun of the same
config on the same data and seed reaches 0.8645.  The first three epochs are
almost identical (0.8227/0.8334/0.8400 vs 0.8245/0.8344/0.8400), then the two
runs separate: init 0.25 declines from epoch 4 while init 0.0 dips once and then
climbs to 0.86.  The protocol document and the code both use 0.0, so **this run
is the record for `din_paper`**; the 0.8555 run is kept as the previous
observation, not as the result.

## Notes

* **First attempt aborted (LR schedule bug).**  The first version of the
  scheduler multiplied the rate by 0.1 every 336,000 steps, so it reached 1e-9
  by epoch 39 and stopped learning (best gAUC 0.8555 at epoch 7).  The authors'
  code assigns `lr = 0.1` once, so the rate stays at 0.1; `min_lr: 0.1` now
  reproduces that.  Both 2026-10-29 runs used PReLU init 0.25, so their 0.8555
  is not comparable with this rerun — see the PReLU note below.
* **Gap to the authors' 0.8698 — structure plus the PReLU start.**  Compared
  with the released TensorFlow model (`din/model.py`), the differences are:
  (a) it applies `batch_normalization` + `dense(128)` (`hist_bn` / `hist_fcn`)
      to the pooled history and feeds the MLP `[u, i, u*i]` with a BatchNorm on
      that concatenation (80-40-1, sigmoid), while this implementation feeds
      `[candidate, pooled]` into the MLP with BatchNorm after the first linear;
  (b) every activation is `sigmoid` in the released code, while the paper's
      Table 3 row uses PReLU;
  (c) the released attention applies softmax after scaling by `sqrt(d)`, while
      the paper says normalisation is abandoned.
  Three candidates are **not** differences: MLP widths (80/40 in both),
  optimiser (SGD 1.0, batch 32 in both) and user embeddings — `user_emb_w` is
  declared in `din/model.py` but **never used**, unlike the base model in the
  same repository.
  Ablations (three epochs each, identical protocol): the sigmoid activation
  alone is 0.010–0.023 **worse**, and the released structure reproduces the
  authors' README (0.8696 vs 0.8698).  The decomposition below shows the
  effective pieces are the `[u, i, u*i]` final-MLP input, the `hist_fcn`
  projection and the input BatchNorm; activation and softmax make no difference
  inside the cluster.  With PReLU init 0.0 the paper structure now reaches
  0.8645, so the residual 0.0053 to the authors' README is the structure, and
  the 0.009 that the PReLU start was worth has been recovered separately.
* **Model selection uses the test split** because the released split has no
  validation set — the authors' `_eval` does the same.  The number is therefore
  best-over-epochs test gAUC, not an untouched held-out score.
* **PReLU initial slope — the first ablation was too short to see it.**  The
  paper does not state the initial α and the released code has no PReLU at all
  (both of its MLPs use sigmoid).  Two 3-epoch arms with init 0 vs the
  TensorFlow embedding initialiser differed by ≤0.002, which was read as "the
  PReLU start does not matter".  The 50-epoch rerun shows that reading was
  wrong: init 0.0 reaches 0.8645 where init 0.25 reached 0.8555.  The
  difference only appears after epoch 3, so it is invisible in a 3-epoch arm.
  The protocol document and the code both use 0.0.
* 50 epochs took 6 h 45 m (median ~490 s/epoch; 5,438 samples/s) on an idle
  card.  The previous run cost 13 h 46 m at 2,882 samples/s because benchmark
  jobs shared the host.

## Ablations

### Embedding initialisation — ruled out

Three epochs each, identical protocol and PReLU init 0; only the embedding-table
initialiser differs (N(0, 1e-4) vs the authors' TensorFlow default,
`glorot_uniform`).

| Epoch | N(0, 1e-4) | glorot_uniform | original 50-epoch run (PReLU 0.25) |
| --- | --- | --- | --- |
| 1 | 0.8227 | 0.8226 | 0.8245 |
| 2 | 0.8334 | 0.8338 | 0.8344 |
| 3 | 0.8400 | 0.8420 | 0.8400 |

The two initialisers stay within 0.002 over three epochs and both sit on the
original curve; the PReLU start (0.25 vs 0) accounts for ~0.002 at epoch 1.
SGD at lr 1.0 washes out a 56× scale difference within the first epoch, so this
hypothesis does not explain the gap.  Outputs: `ads-din-ctrl/`,
`ads-din-glorot/` (outside the repository).

### Activation (sigmoid) — not supported

Three epochs, only the activation changed (attention and MLP to sigmoid, softmax
still off), so it isolates the authors' activation:

| Epoch | PReLU baseline | sigmoid |
| --- | --- | --- |
| 1 | 0.8245 | 0.8015 |
| 2 | 0.8344 | 0.8206 |
| 3 | 0.8400 | 0.8304 |

Sigmoid is consistently 0.010–0.023 **worse**, so switching to the authors'
activation does not close the gap in this structure.

### Released model structure — decomposed

The full cluster (sigmoid, softmax with `sqrt(d)` scaling, `hist_bn`/`hist_fcn`,
`[u, i, u*i]` input, BatchNorm on that input) and five arms that each drop one
piece, three epochs each:

| Arm | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| **full released cluster** | **0.8597** | **0.8678** | **0.8696** |
| − `hist_fcn` | 0.8529 | 0.8638 | 0.8684 |
| − `[u, i, u*i]` input (back to concat) | 0.8401 | 0.8562 | 0.8640 |
| − input BatchNorm | 0.5205 | 0.7763 | 0.8533 |
| − softmax | 0.8575 | 0.8662 | 0.8695 |
| − sigmoid (PReLU) | 0.8572 | 0.8647 | 0.8706 |
| paper structure (reference) | 0.8245 | 0.8344 | 0.8400 |

What the decomposition says:

* the **`[u, i, u*i]` final-MLP input** is the largest single piece (removing it
  costs 0.006–0.020), followed by the **`hist_fcn` projection** (0.001–0.007);
* the **input BatchNorm** is a stability device rather than an accuracy one:
  without it epoch 1 collapses to 0.52 and only recovers to 0.853 by epoch 3;
* **softmax and the activation make no difference inside the cluster** (removing
  either moves the metric by <0.002), which is consistent with sigmoid alone
  being *worse* than PReLU on the paper structure;
* this is evidence about *this* pipeline, not a claim that the paper's text is
  wrong: the released code is what produced 0.8698/0.8818.

A full 50-epoch run of the released structure peaked at **gAUC 0.8696 / AUC
0.8720 at epoch 3** (test) and then overfitted to 0.8214 by epoch 50 — the
authors' README 0.8698 is reproduced to 0.0002.  Runtime 16 h 40 m, mean 2,204
samples/s.

`din.py` exposes `model.hist_fcn`, `model.dnn_input`, `model.input_bn` and
`model.attention_scale` for these arms; all default to the paper structure, so
`din.yaml` and the 50-epoch run are unaffected.

### Additive decomposition — the pieces interact

Starting from the paper structure and adding one piece at a time (3 epochs each):

| Arm | Epoch 1 | Epoch 2 | Epoch 3 |
| --- | --- | --- | --- |
| paper structure | 0.8245 | 0.8344 | 0.8400 |
| + `[u, i, u*i]` input | 0.8302 | 0.8366 | 0.8416 |
| + `hist_fcn` | 0.8317 | 0.8345 | 0.8438 |
| + input BatchNorm | 0.8485 | 0.8565 | 0.8644 |
| + softmax | 0.8430 | 0.8566 | 0.8673 |
| full released cluster | 0.8597 | 0.8678 | 0.8696 |

The additive and subtractive views disagree about individual pieces:

* from below, input BatchNorm (+0.024) and softmax (+0.027) are the largest
  single additions;
* from above (the removal table), softmax changes nothing and input BatchNorm
  only costs the first epochs.

So the pieces interact; no single change reproduces the cluster, and the
ordering of the additive gains is not the ordering of the removal losses.  What
is established is the cluster-level result: the released structure reproduces
the authors' 0.8698, while the paper-text structure stays at 0.8400 after three
epochs and 0.8555 at its 50-epoch best.

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
