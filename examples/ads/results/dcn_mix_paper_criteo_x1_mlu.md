# DCN-Mix — Criteo_x1, paper protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (3 / 3 epochs; best epoch 3) |
| Started / finished | 2026-11-10 12:17:43 → 14:22:11 (CST) |
| Wall clock | 2 h 04 m 28 s (training + validation 7,393.1 s) |
| Config | `examples/ads/conf/train/dcn_mix_paper.yaml` |
| Code commit | `d951e7b` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 6, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 84,800,249 (embedding 39 × 2,086,303 dominates) |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/dcn_mix_paper-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads dcn_mix_paper 6` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; no CPU fallback and no missing MLU
kernel in the launcher's log scan.

## Settings

Paper: Wang et al., *DCN V2*, WWW 2021 (arXiv:2008.13535), Table 6 best setting
for DCN-Mix on Criteo: **LogLoss 0.4408 / AUC 0.8112**.

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `criteo_x1` (reczoo): 13 numeric + 26 categorical, train 33,003,326 / valid 8,250,124 / test 4,587,167 (7:2:1) | — |
| model | parallel DCN-V2: CrossNetMix ‖ DNN, concatenated into one logit | `[paper]` Fig. 1(b) |
| embedding size | 39 | `[paper]` §7.1.3 |
| cross layers | 3 | `[paper]` Table 6 (`ll=3`) |
| experts | 4 | `[paper]` Table 6 (`K=4`) |
| rank per expert | 64 | `[paper]` Table 6 |
| cross activation | hard_tanh | `[paper]` Table 6 |
| gate | constant (`G_i = 1/K`) | `[paper]` Table 6 |
| kernel init | He Normal | `[paper]` §7.1.3 |
| DNN | [512, 512], ReLU | `[paper]` Table 6 / §3.3 |
| dropout | 0 | `[ours]` |
| optimizer | Adam, lr 5e-4 | `[paper]` / `[ours]` (top of the 1e-4–5e-4 range) |
| gradient clipping | global norm 10 | `[paper]` §7.1.3 |
| EMA decay | 0.9999, evaluated and exported | `[paper]` §7.1.3 |
| weight decay / λ | 0 (Adam); embedding L2 1e-5 | `[ours]` |
| batch size | 512 | `[paper]` §7.1.3 |
| epochs | 3 (paper: 150 k steps ≈ 2.3 epochs here) | `[paper]` / `[ours]` |
| early stopping | none (fixed step budget) | `[paper]` |
| metric | AUC / logloss | `[paper]` Table 6 |
| seed | 2021 | `[ours]` |

The paper's day-based split is not distributed; `criteo_x1` is the reczoo 7:2:1
split, so the paper's 0.8112 is context.  The comparable reference on this split
is BARS full-rank DCN-V2: AUC 0.8142 / logloss 0.4378.

## Curve

33.0 M rows / batch 512 = 64,460 steps per epoch.

| Epoch | Train loss | Valid AUC | Valid logloss | Epoch time |
| --- | --- | --- | --- | --- |
| 1 | 0.4619 | 0.8071 | 0.4446 | 2,721.2 s |
| 2 | 0.4569 | 0.8093 | 0.4423 | 2,310.8 s |
| **3 (best)** | **0.4559** | **0.8101** | **0.4415** | 2,319.0 s |

All three epochs improve on validation AUC, i.e. the 3-epoch budget truncates a
run that has not flattened yet, as on the P800.

## Results

| Metric | This run (best epoch 3) | P800 same config | BARS DCN-V2 (same split) | Paper (different split) |
| --- | --- | --- | --- | --- |
| test AUC | 0.8104 | 0.8104 | 0.8142 | 0.8112 |
| test logloss | 0.4411 | 0.4411 | 0.4378 | 0.4408 |
| valid AUC | 0.8101 | 0.8102 | — | — |

The MLU run reproduces the P800 run of the same config to within 0.00003 AUC
and 0.00003 logloss.

## Notes

* Throughput: **14,232 samples/s median** (epoch 1 12,128, epochs 2–3 14,282 /
  14,232) vs ~9.5 k samples/s on the P800, so the MLU is ~1.5× faster here.
* Card 6 was shared with another user's job (19 GB).  Epoch 1 took 2,721 s and
  epochs 2–3 ~2,315 s; warm-up plus that sharing is the likely cause.  The test
  metric is unaffected.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; not sampled while training.
* The run uses the 2-D-matmul `CrossNetMix` introduced in `929e39a`; no vendor
  JIT failure occurred on MLU (the failure described in the P800 page was
  specific to that stack).

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation-AUC epoch (3) weights, 324 MB |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | written at the end: summary + full history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads dcn_mix_paper 6
```
