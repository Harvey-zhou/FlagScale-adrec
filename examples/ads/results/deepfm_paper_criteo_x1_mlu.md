# DeepFM — Criteo_x1, paper protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 6; best epoch 1) |
| Started / finished | 2026-11-10 12:17:44 → 13:02:36 (CST) |
| Wall clock | 44 min 52 s (training + validation 2,636.8 s) |
| Config | `examples/ads/conf/train/deepfm_paper.yaml` |
| Code commit | `d951e7b` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 4, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 23,427,077 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/deepfm_paper-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads deepfm_paper 4` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; no CPU fallback and no missing MLU
kernel in the launcher's log scan.

## Settings

Paper: Guo et al., *DeepFM*, IJCAI 2017.

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `criteo_x1` (reczoo): 13 numeric + 26 categorical fields; train 33,003,326 / valid 8,250,124 / test 4,587,167 (7:2:1, AFN split) | — |
| model | FM (order-1 + order-2 inner products) + DNN over the same field embeddings | `[paper]` §2.1 |
| embedding / FM latent dim | 10 | `[paper]` §3.1.4 |
| DNN | 400-400-400, ReLU | `[paper]` §3.1.4 |
| dropout | 0.5 | `[paper]` §3.1.4 |
| BatchNorm | off | `[ours]` the paper describes no BatchNorm (BARS adds it) |
| optimizer | Adam | `[paper]` §3.1.4 |
| learning rate | 1e-3, constant | `[ours]` the paper states no lr |
| weight decay / embedding L2 | 0 / 0 | `[ours]` the paper uses dropout only |
| gradient clipping | none | `[ours]` |
| batch size | 4096 | `[ours]` not in the paper; Criteo convention |
| epochs | 100 (cap) | `[ours]` |
| early stopping | stop after 5 epochs without validation AUC improvement | `[ours]` the paper gives no rule |
| metric | AUC / logloss | `[paper]` §3.1.2 |
| seed | 2021 | `[ours]` |

## Data-split caveat

The paper uses a random 90/10 split of the 45M-row Kaggle Criteo file without
publishing the split or a seed.  `criteo_x1` is the reczoo release with a 7:2:1
split (following AFN, AAAI 2020), so the paper's 0.8007 / 0.45083 is context,
not a reproducible target.  The comparable reference on this split is the
BARS/FuxiCTR `deepfm.yaml` run: AUC 0.8137 / logloss 0.4381.

## Curve

| Epoch | Train loss | Valid AUC | Valid logloss | Epoch time |
| --- | --- | --- | --- | --- |
| **1 (best)** | **0.4543** | **0.8034** | **0.4482** | 584.7 s |
| 2 | 0.4374 | 0.8016 | 0.4509 | 575.6 s |
| 3 | 0.4255 | 0.7981 | 0.4567 | 572.1 s |
| 4 | 0.4157 | 0.7930 | 0.4650 | 553.5 s |
| 5 | 0.4078 | 0.7895 | 0.4742 | 150.7 s |
| 6 | 0.4016 | 0.7852 | 0.4823 | 151.5 s |

Early stopping fired at epoch 6: the best validation AUC was epoch 1 and nothing
improved for the next 5 epochs.  As on the P800, training loss keeps falling
while validation AUC falls every epoch — the paper configuration overfits from
the first epoch on this 33M-row training set.

## Results

| Metric | This run (best epoch 1) | P800 same config | BARS `deepfm.yaml` | Paper (different split) |
| --- | --- | --- | --- | --- |
| test AUC | 0.8039 | 0.8040 | 0.8137 | 0.8007 |
| test logloss | 0.4477 | 0.4477 | 0.4381 | 0.45083 |
| valid AUC | 0.8034 | 0.8036 | — | — |

The MLU run reproduces the P800 run of the same config to within 0.00003 AUC
and 0.00003 logloss, so the backend does not change the result.

## BF16

Full run of the same config with `train.model.torch_dtype=bf16`
(`2026-11-13 10:38:55 → 11:02:50`, card 2, 6 epochs, early stop at 6, best
epoch 1 — the same schedule as FP32).  Autocast: weights and optimizer state
stay fp32, and the run wrote to a separate `ADREC_OUT_ROOT` (`out-bf16`).

| Metric | FP32 (best epoch 1) | BF16 (best epoch 1) | Δ |
| --- | --- | --- | --- |
| test AUC | 0.803867 | 0.803764 | −0.000103 |
| test logloss | 0.447683 | 0.447897 | +0.000214 |
| best valid AUC | 0.80343 | 0.80335 | −0.00008 |
| epochs run | 6 | 6 | |

Both curves overfit identically: the per-epoch valid AUC agrees to ≤ 0.001 at
every epoch, and the final gap is 1.0e-4 AUC / 2.1e-4 logloss — the same order
as the BARS DeepFM pair, i.e. BF16 is accuracy-neutral here too.

Throughput is the awkward one, because the FP32 baseline was taken on a shared
card (see the note above): its full-run median is 58,655 samples/s but its two
clean epochs ran at ~218 k.  BF16 held 162–178 k samples/s for all six epochs
(median **173,151**), so against the clean FP32 rate it is **~20 % slower**,
consistent with the BARS DeepFM probes (87,758 vs 99,295).  The 43.9 min → 19.9
min wall-clock drop is the co-tenant leaving, not BF16.

## Notes

* **Card sharing affects throughput, not the metric.**  Card 4 was shared with
  another user's job for the first part of the run (`cnmon` showed a 34 GB
  process at 12:22).  Epochs 1–4 ran at 56–60 k samples/s; epochs 5–6, after
  that job's activity dropped to 0 %, ran at ~218 k samples/s.  The samples/s
  figure is therefore a range (56 k–219 k), not a clean single-card number.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; not sampled while training.
* The best epoch is saved and the test pass runs on it, as on the P800.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best validation-AUC epoch (1) weights, 90 MB |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/metrics.json` | written at the end: summary + full history |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads deepfm_paper 4
```
