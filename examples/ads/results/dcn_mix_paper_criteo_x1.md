# DCN-Mix — Criteo_x1, paper protocol

| Field | Value |
| --- | --- |
| Status | **queued** — first attempt died on a vendor JIT error before epoch 1; fixed build waits for a free card |
| Config | `examples/ads/conf/train/dcn_mix_paper.yaml` |
| Code commit | `ef9a5d6` |
| Device | Kunlunxin P800, single card, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 84,800,249 (embedding 39 × 2,086,303 dominates) |
| Output | `/data2/jiangxuhui/logs/ads-dcnmix-paper/` (to be produced) |

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
CPU smoke run passes; the GPU build is queued on the first card that frees.

## Results

Pending.  This section will hold the epoch curve and the comparison with the
BARS reference and the paper.
