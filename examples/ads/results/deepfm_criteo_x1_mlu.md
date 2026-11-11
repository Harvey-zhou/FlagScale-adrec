# DeepFM — Criteo_x1, BARS/FuxiCTR protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 15; best epoch 13) |
| Started / finished | 2026-11-11 10:26:41 → 12:36:30 (CST) |
| Wall clock | 2 h 09 m 49 s (training 7,690.5 s) |
| Config | `examples/ads/conf/train/deepfm.yaml` |
| Code commit | `08fa8dd` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 3, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 23,429,477 (identical shape to the BARS model) |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/deepfm-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads deepfm 3` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

This is the BARS/FuxiCTR reference config
(`ranking/ctr/DeepFM/DeepFM_criteo_x1`, FuxiCTR v1.1.0), not the paper protocol
of [`deepfm_paper_criteo_x1_mlu.md`](deepfm_paper_criteo_x1_mlu.md).

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `criteo_x1`: 13 numeric + 26 categorical fields; train 33,003,326 / valid 8,250,124 / test 4,587,167 rows (7:2:1) | BARS |
| model | FM (order-1 + order-2) ‖ DNN over the same field embeddings | BARS |
| embedding | 10 | BARS |
| DNN | [400, 400, 400], ReLU, BatchNorm, dropout 0.1 | BARS |
| optimizer | Adam, lr 1e-3 with reduce-on-plateau (observed 1e-3 → 1e-4 at epoch 11 → 1e-5 at epoch 15) | BARS |
| embedding regularizer | 1e-5 | BARS |
| gradient clipping | global norm 10 | BARS |
| batch size | 4096 (8,058 steps per epoch) | BARS |
| epochs / early stop | 100 cap, patience 2 on valid AUC, best checkpoint restored for test | BARS |
| seed | 2021 | BARS |

The validation split (8.25 M rows) drives early stopping; the reported number is
on the held-out test split, as in BARS.

## Curve

| Epoch | LR | Train loss | Valid AUC | Valid logloss | Epoch time |
| --- | --- | --- | --- | --- | --- |
| 1 | 1e-3 | 0.4643 | 0.8030 | 0.4484 | 332.4 s |
| 2 | 1e-3 | 0.4586 | 0.8056 | 0.4460 | 337.0 s |
| 3 | 1e-3 | 0.4571 | 0.8067 | 0.4448 | 344.9 s |
| 4 | 1e-3 | 0.4563 | 0.8072 | 0.4442 | 337.2 s |
| 5 | 1e-3 | 0.4559 | 0.8077 | 0.4438 | 320.2 s |
| 6 | 1e-3 | 0.4555 | 0.8080 | 0.4436 | 337.1 s |
| 7 | 1e-3 | 0.4553 | 0.8083 | 0.4432 | 371.1 s |
| 8 | 1e-3 | 0.4551 | 0.8085 | 0.4430 | 774.5 s |
| 9 | 1e-3 | 0.4549 | 0.8087 | 0.4429 | 782.8 s |
| 10 | 1e-3 | 0.4547 | 0.8087 | 0.4429 | 775.7 s |
| 11 | 1e-4 | 0.4428 | 0.8127 | 0.4392 | 775.4 s |
| 12 | 1e-4 | 0.4384 | 0.8132 | 0.4387 | 748.7 s |
| **13 (best)** | **1e-4** | **0.4366** | **0.8134** | **0.4386** | 695.0 s |
| 14 | 1e-4 | 0.4353 | 0.8133 | 0.4387 | 320.6 s |
| 15 | 1e-5 | 0.4308 | 0.8128 | 0.4395 | 320.5 s |

Early stop fired at epoch 15: the best valid AUC was epoch 13 and epochs 14–15
did not improve on it.  The best checkpoint was restored before the test pass.

## Results

| Metric | This run (best epoch 13) | BARS published | Δ |
| --- | --- | --- | --- |
| test AUC | **0.813723** | **0.813666** | +0.000057 |
| test logloss | **0.438119** | **0.438147** | −0.000028 |

The MLU run matches the BARS reference to 6e-5 AUC / 3e-5 logloss, i.e. the
backend does not change the result.  The paper-protocol run on the same split
(`deepfm_paper`) is 0.0098 AUC lower because it uses dropout 0.5, no BatchNorm
and no embedding regularizer.

## Notes

* Throughput: **95,699 samples/s median** (42,161–103,058 over the 15 epochs).
  Epochs 8–13 ran at 42–47 k samples/s — about half the rate of the other
  epochs — while epochs 14–15 returned to ~103 k.  That is the signature of a
  co-tenant on card 3 during that window; the metric is unaffected.
* The LR plateaus at 1e-3 for ten epochs (AUC creeps from 0.803 to 0.809), then
  the 10× drop to 1e-4 is what produces the jump to 0.813; the second drop to
  1e-5 at epoch 15 costs 0.0005 AUC and triggers the stop.
* The final test pass took 3.1 s.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; the clock was not sampled while training.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best valid-AUC epoch (13) weights |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads deepfm 3
```
