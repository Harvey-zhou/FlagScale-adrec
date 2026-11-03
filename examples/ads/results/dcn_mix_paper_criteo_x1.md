# DCN-Mix — Criteo_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (3 / 3 epochs; best epoch 3) |
| Started / finished | 2026-11-02 15:50:51 → 18:46:05 (CST) |
| Wall clock | 2 h 55 m (3 epochs of 3,443 / 3,509 / 3,383 s + 156 s of validation) |
| Config | `examples/ads/conf/train/dcn_mix_paper.yaml` |
| Code commit | `ef9a5d6` (the 2-D-matmul `CrossNetMix`; `929e39a` only records the protocol); the config's `log_every: 1000` was added while the run was already going and only affects progress printing |
| Device | Kunlunxin P800, card 3, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 84,800,249 (embedding 39 × 2,086,303 dominates) |
| Output | `/data2/jiangxuhui/logs/ads-dcnmix-paper/ckpts/dcn_mix_paper-cuda/checkpoint/` |
| Command | `launch-ads-paper.sh 3 dcn_mix_paper /data2/jiangxuhui/logs/ads-dcnmix-paper` |

## Settings

Paper: Wang et al., *DCN V2*, WWW 2021 (arXiv:2008.13535), Table 6 best setting
for DCN-Mix on Criteo: **LogLoss 0.4408 / AUC 0.8112**.

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `criteo_x1` (reczoo): 13 numeric + 26 categorical, train 33,003,326 / valid 8,250,124 / test 4,587,167 (7:2:1) | — |
| model | parallel DCN-V2: CrossNetMix ‖ DNN, concatenated into one logit | `[paper]` Fig. 1(b) |
| embedding size | 39 | `[paper]` §7.1.3 ("39 for Criteo") |
| cross layers | 3 | `[paper]` Table 6 (`ll=3`) |
| experts | 4 | `[paper]` Table 6 (`K=4`) |
| rank per expert | 64 (paper: `r=258` total over 4 experts ≈ 64.5) | `[paper]` Table 6 |
| cross activation | hard_tanh | `[paper]` Table 6 ("hard_tanh activation for DCN-Mix") |
| gate | constant (`G_i = 1/K`) | `[paper]` Table 6 ("gate as constant") |
| kernel init | He Normal | `[paper]` §7.1.3 |
| DNN | [512, 512], ReLU | `[paper]` Table 6 ("(2, 512)") / §3.3 |
| dropout | 0 | `[ours]` the paper mentions none |
| optimizer | Adam, lr 5e-4 | `[paper]` (Adam) / `[ours]` (within the 1e-4–5e-4 narrowed range) |
| gradient clipping | global norm 10 | `[paper]` §7.1.3 |
| EMA decay | 0.9999, evaluated and exported | `[paper]` §7.1.3 |
| weight decay / λ | 0 (Adam); embedding L2 1e-5 | `[ours]` the paper searches λ ∈ {0, 3e-5, 1e-4} |
| batch size | 512 | `[paper]` §7.1.3 |
| epochs | 3 (paper: 150k steps ≈ 2.3 epochs here) | `[paper]` §7.1.3 / `[ours]` |
| early stopping | none (fixed step budget) | `[paper]` |
| metric | AUC / logloss | `[paper]` Table 6 |
| seed | 2021 | `[ours]` |

The paper's split (first 6 days train, last day split into valid/test) is not
distributed; this dataset is the reczoo 7:2:1 split, so the paper's 0.8112 is
context rather than a target.  The comparable reference on this split is the
BARS full-rank DCN-V2: AUC 0.8142 / logloss 0.4378 (mixture off).

## First attempt: vendor JIT failure

The run died before finishing epoch 1 while compiling a batched-matmul kernel:

```
1 error generated when compiling for xpu3.
... bmm_one_loop ... could not match ...
```

`CrossNetMix` used 3-D `torch.matmul` (bmm).  It now uses only 2-D matmuls
(`x @ V`, `v @ Cᵀ`, `v @ Uᵀ`, then a weighted sum over experts), which is
algebraically identical to Eq. (3) and avoids the failing kernel family.  The
fix is committed in `929e39a`; the rerun on card 3 (15:50) got past the JIT
step and trained normally, so the workaround is confirmed on this stack.

## Curve

33.0 M rows / batch 512 = 64,460 steps per epoch, ~9.5 k samples/s.

| Epoch | Train loss | Valid AUC | Valid logloss | Epoch time |
| --- | --- | --- | --- | --- |
| 1 | 0.4619 | 0.8071 | 0.4445 | 3,443 s |
| 2 | 0.4569 | 0.8094 | 0.4423 | 3,509 s |
| **3 (best)** | **0.4559** | **0.8102** | **0.4415** | 3,383 s |

All three epochs improve on validation AUC, i.e. the 3-epoch budget (~193 k
steps) truncates a run that has not flattened yet.  The paper's "150 k steps"
converts to ~2.3 epochs on this split; we rounded up to 3.

## Results

| Metric | This run (best epoch 3) | BARS `DCNv2_criteo_x1` (same split) | Paper (different split) |
| --- | --- | --- | --- |
| test AUC | 0.810421 | 0.814195 | 0.8112 |
| test logloss | 0.441089 | 0.437788 | 0.4408 |
| valid AUC | 0.810169 | — | — |

0.0038 AUC below the BARS reference and 0.0033 logloss above it; the paper's
numbers come from the day-based split and are context only.

**What differs from BARS is mostly not paper-specified**: BARS uses
embedding size 10, low-rank 32, a 500-500-500 DNN, dropout 0.1, Adam 1e-3 and
batch 4096, while the paper's best setting (which we followed) is embedding 39,
low-rank 64 per expert, a 512-512 DNN, Adam 5e-4 and batch 512.  The remaining
gap has not been decomposed, and no ablation has been run.  The cheapest
informative check is the learning rate: the paper narrows Adam to 1e-4–5e-4 and
we took the top of that range, while BARS uses 1e-3.

## Notes

* The best checkpoint is epoch 3 and the test pass runs on it.
* Training was ~9.5 k samples/s, so one epoch (~64.5 k steps at batch 512)
  costs ~1 h.  The card was shared with other users for part of the run; the
  three epoch times are within 4 % of each other, so the sharing did not
  distort the comparison.
* EMA (decay 0.9999) was applied and the EMA weights are what got evaluated, as
  the paper specifies; this is the same code path as the BARS config.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation-AUC epoch weights (339 MB) |
| `checkpoint/history.json` | per-epoch train/valid history, written incrementally |
| `checkpoint/metrics.json` | written at the end: summary + full history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=dcn_mix_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
