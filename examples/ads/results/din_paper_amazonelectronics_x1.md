# DIN — Amazon Electronics_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (all 50 epochs) |
| Started / finished | 2026-10-29 10:06:07 → 23:52 (CST) |
| Wall clock | 13 h 46 m (training 49,547 s) |
| Config | `examples/ads/conf/train/din_paper.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | launched from the uncommitted working tree at 10:06, committed 11:01 as `8211572`. The run predates two later edits: the 10:58 write-then-rename of `history.json` (no effect on training) and `e463e1e` (PReLU slope `init=0`), which is **not** in this run — it used PyTorch's default 0.25. |
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

The learning rate drops from 1.0 to 0.1 at step 336,000 and then stays at 0.1.
The drop lands at 336,000 / 81,524 ≈ 4.12 epochs, i.e. **inside epoch 5**.  The
`LR` column below is the rate at the *start* of the epoch, which is why epoch 5
still shows 1.0 and epoch 6 is the first one that starts at 0.1.

gAUC peaks at epoch 7 and declines afterwards — the run continues to epoch 50
because that is what the authors' code does, and the best checkpoint is saved at
every improvement.

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

### Against the available targets

| Target | gAUC | This run (paper structure) |
| --- | --- | --- |
| Paper Table 3 (DIN, PReLU) | 0.8818 | −0.0263 |
| Authors' README (their own run of this code) | 0.8698 | −0.0143 |
| BARS/FuxiCTR `din.yaml` on this data (fresh run, batch 1024) | 0.8823 | −0.0280 |
| Released DIN structure, this repo (50 epochs) | 0.8696 | −0.0002 vs the README |

The BARS figure is from a fresh run of `din.yaml` on this tree (9 epochs, test
AUC 0.8846 / logloss 0.4296); the older `flagscale-adrec-20261014` snapshot
reported 0.8835, within run-to-run variation.

The authors' own README reports 0.8698 with the released code on the same kind
of split, so the paper's 0.8818 is not reproducible as published even by its
authors; 0.8698 is the more meaningful target, and this run is 0.014 short of
it.

## Notes

* **First attempt aborted (LR schedule bug).**  The first version of the
  scheduler multiplied the rate by 0.1 every 336,000 steps, so it reached 1e-9
  by epoch 39 and stopped learning (best gAUC 0.8555 at epoch 7).  The authors'
  code assigns `lr = 0.1` once, so the rate stays at 0.1; `min_lr: 0.1` now
  reproduces that.  Up to epoch ~10 both runs are identical (same seed), which is
  why the best number is unchanged — the model overfits after epoch 7 either way.
* **Gap to the authors' 0.8698 — the structure explains it (verified by ablation).**  Compared
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
  Ablations (three epochs each, identical protocol): the embedding
  initialisation makes no difference (≤0.002), the sigmoid activation alone is
  0.010–0.023 **worse**, and the released structure reproduces the authors'
  README (0.8696 vs 0.8698).  The decomposition below shows the effective
  pieces are the `[u, i, u*i]` final-MLP input, the `hist_fcn` projection and
  the input BatchNorm; activation and softmax make no difference inside the
  cluster.
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
hypothesis does not explain the 0.014 gap.  Outputs: `ads-din-ctrl/`,
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
