# DIN paper protocol — Amazon Electronics

How `examples/ads/conf/train/din_paper.yaml` (PReLU) and `din_paper_dice.yaml`
(Dice) are built, and why every number is what it is.

The two other configs in this directory are **not** paper protocols: `din.yaml`
reproduces the BARS/FuxiCTR v2.0.1 benchmark (`DIN_amazonelectronics_x1`, test
gAUC 0.8835).  This document describes the separate attempt to reproduce the
original KDD'18 paper.

* Paper: Zhou et al., *Deep Interest Network for Click-Through Rate Prediction*,
  KDD 2018, arXiv:1706.06978v4.
* Authors' code: <https://github.com/zhougr1993/DeepInterestNetwork>
  (`din/train.py`, `din/model.py`, `din/Dice.py`).
* Dataset: reczoo/AmazonElectronics_x1, the output of the authors'
  `din/build_dataset.py` (see [EVIDENCE.md](EVIDENCE.md)).

Every value below is tagged with where it comes from:

| Tag | Meaning |
| --- | --- |
| `[paper]` | stated in the paper; must not change in a paper-protocol run |
| `[code]` | not in the paper; taken from the authors' released code |
| `[ours]` | neither source specifies it; our choice, free to tune |

## 1. Data and split

The authors' split has only train and test (`data.valid_split: test`):

* one row per `(user, target item)` pair, with the target ranked against one
  random negative (seed 1234 in the authors' code);
* train: for every user, targets `k = 1 … n-2` → **2,608,764 rows**
  (1,304,382 positives + 1,304,382 negatives);
* test: target `n` given the first `n-1` behaviours → **384,806 rows**
  (192,403 positives + 192,403 negatives);
* 192,403 users, item vocabulary 63,003, category vocabulary 803 (paper §6.1
  says 192,403 users / 63,001 goods / 801 categories; the extra two ids are the
  zero padding row and the reczoo index origin, documented in EVIDENCE.md);
* behaviour history is truncated/padded to `max_len = 100`;
* features are exactly the paper's: `item_id`, `cate_id`, `item_history`,
  `cate_history` (paper §6.1). `user_id` is only the gAUC grouping key, never a
  model input — the authors' code has a user embedding, the paper's feature list
  does not.

## 2. Metric

`monitor: gAUC` — the user-weighted AUC of paper §6.3 Eq. (10), implemented in
`flagscale/train/metrics/ads.py` (`group_auc`).  With two impressions per user it
is the mean per-user AUC.

## 3. Hyperparameters

| Setting | Value | Source | Config key |
| --- | --- | --- | --- |
| optimizer | SGD | `[paper]` §6.1 "we use SGD as the optimizer" | `optimizer.name: sgd` |
| momentum | 0.0 | `[code]` plain `GradientDescentOptimizer` | `optimizer.momentum` |
| learning rate | 1.0 | `[paper]` §6.1 "learning rate starts at 1" | `optimizer.lr` |
| decay rate | 0.1 | `[paper]` §6.1 | `scheduler.gamma` |
| decay schedule | ×0.1 at step 336,000, then constant 0.1 | `[code]` `if global_step % 336000 == 0: lr = 0.1` | `scheduler.step_size_steps`, `scheduler.min_lr` |
| warmup | none | `[ours]` paper does not use warmup | `scheduler.warmup_steps: 0` |
| batch size | 32 | `[paper]` §6.1 | `batch_size` |
| epochs | 50 | `[code]` `for _ in range(50)`, no early stop | `epochs` |
| early stopping | disabled | `[ours]` paper gives none; the code runs all 50 | `early_stop_patience: 0` |
| loss | negative log-likelihood (BCE) | `[paper]` Eq. (2) | `ctr_loss` |
| gradient clipping | global norm 5 | `[code]` `tf.clip_by_global_norm(gradients, 5)` | `optimizer.max_gradient_norm` |
| weight decay | 0.0 | `[ours]` paper applies MBA (λ=0.01) only on Alibaba §5.1 | `optimizer.weight_decay` |
| embedding L2 | 0.0 | `[ours]` same reason; §6.5 says Amazon has no severe overfitting | `optimizer.embedding_regularizer` |
| dropout | 0.0 | `[ours]` dropout is an Alibaba ablation §6.5 | `model.net_dropout`, `attention_dropout` |
| seed | 1234 | `[code]` `random/np/tf.set_random_seed(1234)` | `seed` |
| precision | fp32 | `[ours]` the paper mentions no mixed precision | `model.torch_dtype` |
| embedding dim | 64 item + 64 category | `[code]` `item_emb_w` 64 and `cate_emb_w` 64 | `model.embedding_dim` |
| attention MLP | [80, 40] → 1 | `[code]` `dense(80) → dense(40) → dense(1)` | `model.attention_hidden_units` |
| final MLP | [80, 40] → 1 | `[code]` `dense(80) → dense(40) → dense(1)` | `model.dnn_hidden_units` |
| activation | PReLU (Dice config: Dice) | `[paper]` Table 3 footnote / §5.2 | `attention_hidden_activations`, `dnn_activations` |
| BatchNorm | on | `[code]` `tf.layers.batch_normalization` | `model.batch_norm` |
| attention softmax | **off** | `[paper]` §4.3 "normalization with softmax … is abandoned" | `model.din_use_softmax: false` |
| Dice epsilon | 1e-8 | `[paper]` §5.2 ("set to be 1e-8 in our practice") | `model.activation_epsilon` |

Model size with this configuration: **4,152,226 parameters**
(embeddings 4,083,584; attention MLP 44,321; final MLP 23,841; plus BatchNorm
and PReLU parameters).  The BARS config `din.yaml` has 5,403,522 parameters —
different MLP widths, softmax and dropout, as expected.

### Learning-rate timeline on this dataset

2,608,764 / 32 = 81,524 steps per epoch, so 336,000 steps is **epoch 4.12**:

```
step 0                    lr = 1.0
step 335,999              lr = 1.0
step 336,000 (epoch 5)    lr = 0.1
step 4,076,200 (epoch 50) lr = 0.1
```

`min_lr: 0.1` is what reproduces the authors' behaviour.  Their code assigns
`lr = 0.1` (an absolute value), so the rate stays at 0.1; a plain geometric
schedule `1.0 → 0.1 → 0.01 → …` collapses to 1e-9 by epoch 39 and stops
learning.  The first run of this protocol made exactly that mistake; its best
gAUC was 0.8555 at epoch 7 and the remaining 30+ epochs were wasted.

## 4. Training loop

`flagscale/train/train_ads.py`, mode `scheduler.type: step`:

1. load the whole split into host memory, shuffle every epoch with a seeded
   sampler, form batches of 32;
2. per step: set the learning rate from the schedule, forward, BCE, backward,
   clip grad norm to 5, SGD step;
3. after every epoch: score the test split in batches of 8192, record
   `gAUC/AUC/logloss`, and if the gAUC is a new best, save `model.pt` and
   `history.json` immediately (a 15-hour run that dies still leaves the best
   model behind);
4. after the last epoch: restore the best weights, score them once more, write
   `metrics.json` with the optimizer/scheduler settings and the full history.

The fixed schedule never changes the learning rate on a plateau; `plateau`
remains the default mode for the BARS configs, where a non-improvement still
multiplies the rate by 0.1 and `early_stop_patience` counts epochs.

## 5. Paper/code conflicts, disclosed

1. **Softmax.**  §4.3 says softmax normalisation is abandoned (to keep the
   intensity of user interest); the released code still applies
   `tf.nn.softmax` after dividing by `sqrt(d)`.  We follow the paper: no softmax.
2. **Activation.**  Table 3 says all deep models use PReLU; the released code
   uses `sigmoid` in both MLPs.  We follow the paper (PReLU) for `din_paper`
   and the paper's Dice variant for `din_paper_dice`.
3. **Final MLP input.**  The code concatenates `[u, i, u*i]`; paper Fig. 2
   concatenates the user representation with the candidate embedding.  This
   implementation feeds `[candidate_embedding, pooled_history]`.
4. **Test-set model selection.**  The authors' split has no validation set, so
   the per-epoch test gAUC also selects the checkpoint — exactly what the
   authors' `_eval` does.  The reported number is therefore a best-over-epochs
   test score, not an untouched held-out score.
5. **Mini-batch aware regularisation.**  §5.1/§6.5 describe it for Alibaba
   (λ = 0.01); it is not applied here because the paper does not use it on
   Amazon and the released code has no regulariser.
6. **Number of runs.**  The paper averages 5 runs and notes the initialisation
   affects AUC by less than 0.0002.  We run one seed (1234).
7. **PReLU initial slope.**  Slopes start at 0 (`nn.PReLU(width, init=0.0)`, so
   the unit begins as a plain ReLU), following the form of paper Eq. (8).  PyTorch's
   default is 0.25.  The released code has no PReLU to copy (both MLPs use
   `sigmoid`); only its Dice `alpha` starts at 0.
8. **Single card only.**  The step schedule counts optimizer steps per process.
   Under DDP each rank reaches step 336,000 after consuming `world_size` times
   the samples of a single card, so the learning-rate drop comes later than in
   the single-card protocol.  Batch 32 is a single-card setting; do not use
   `din_paper*` with more than one card.
9. **Embedding initialisation.**  The tables are initialised N(0, 1e-4)
   (FuxiCTR's convention, used by every BARS result here), while the authors'
   `tf.get_variable` tables get TensorFlow's default `glorot_uniform`
   (bound ~0.0098 for a 63,003 x 64 table, ~56x the standard deviation).  The
   paper states neither.  `model.embedding_init: normal | glorot_uniform` now
   selects it; the finished 50-epoch run used `normal` and a 3-epoch
   `glorot_uniform` control was started to test whether this explains the gap.

## 6. How to run

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets        # contains amazonelectronics_x1
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

# single card, vendor torch, 50 epochs
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=din_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1

# Dice variant
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=din_paper_dice train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```

On the Kunlunxin P800 host the helper
`examples/run-adrec-kunlunxin.sh ads din_paper <card>` adds the vendor
environment and a health preflight.  Outputs go to
`$FLAGOS_ADS_CKPTS/<name>-cuda/checkpoint/{model.pt,history.json,metrics.json,config.json}`.

A smoke run that takes a minute:

```bash
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=din_paper train.hardware.device_type=cpu \
  train.epochs=1 train.data.max_samples=4096 train.data.eval_max_samples=2048
```

## 7. Measured cost

| Device | Throughput | Epoch | Full 50 epochs |
| --- | --- | --- | --- |
| Kunlunxin P800 (vendor torch 2.9, fp32, batch 32) | ~2,480 samples/s | ~1,050 s (17.5 min) | ~14.6 h |

Batch 32 is set by the paper and leaves the accelerator badly under-utilised on
purpose: changing it would change the protocol.

## 8. Results

Paper targets (Table 3, Amazon Electronics, user-weighted AUC):

| Model | Paper | Authors' README (gAUC) |
| --- | --- | --- |
| DIN (PReLU) | 0.8818 | 0.8698 |
| DIN with Dice | 0.8871 | 0.8711 |

Results of `din_paper` (PReLU) and `din_paper_dice` on this implementation are
recorded next to the checkpoints (`metrics.json`) and summarised in
[EVIDENCE.md](EVIDENCE.md) once the runs finish.
