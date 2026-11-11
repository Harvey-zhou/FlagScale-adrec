# DCN-Mix — Criteo_x1, BARS/FuxiCTR protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 18; best epoch 16) |
| Started / finished | 2026-11-11 10:26:41 → 12:01:36 (CST) |
| Wall clock | 1 h 34 m 55 s (training 5,606.5 s) |
| Config | `examples/ads/conf/train/dcn_mix.yaml` |
| Code commit | `08fa8dd` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 4, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 21,875,089 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/dcn_mix-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads dcn_mix 4` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

This is the BARS/FuxiCTR reference config
(`ranking/ctr/DCNv2/DCNv2_criteo_x1`, FuxiCTR v1.1.0) with
`use_low_rank_mixture` switched on, not the paper protocol of
[`dcn_mix_paper_criteo_x1_mlu.md`](dcn_mix_paper_criteo_x1_mlu.md).

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `criteo_x1`: 13 numeric + 26 categorical; train 33,003,326 / valid 8,250,124 / test 4,587,167 (7:2:1) | BARS |
| model | parallel DCN-V2: CrossNetMix ‖ DNN, concatenated into one logit | BARS / paper Fig. 1(b) |
| embedding | 10 | BARS |
| cross layers | 3 | BARS |
| low-rank mixture | on: rank 32, 4 experts | BARS config (`use_low_rank_mixture=True`) |
| parallel DNN | [500, 500, 500], ReLU, dropout 0.1, no BatchNorm | BARS |
| optimizer | Adam, lr 1e-3 with reduce-on-plateau (observed 1e-3 → 1e-4 at epoch 15 → 1e-5 at epoch 18) | BARS |
| embedding regularizer | 1e-5 | BARS |
| gradient clipping | global norm 10 | BARS |
| batch size | 4096 (8,058 steps per epoch) | BARS |
| epochs / early stop | 100 cap, patience 2 on valid AUC, best checkpoint restored for test | BARS |
| seed | 2021 | BARS |

BARS publishes the **full-rank** DCN-V2 only (0.814195 / 0.437788); the mixture
is switched on here because the config is the DCN-Mix one.  The paper's own
comparison (Table 6) has Mix 0.0003 AUC *below* full-rank V2, so a small
negative delta against the BARS number is expected.

## Curve

| Epoch | LR | Train loss | Valid AUC | Valid logloss | Epoch time |
| --- | --- | --- | --- | --- | --- |
| 1 | 1e-3 | 0.4609 | 0.8047 | 0.4469 | 311.7 s |
| 2 | 1e-3 | 0.4548 | 0.8071 | 0.4444 | 312.8 s |
| 3 | 1e-3 | 0.4532 | 0.8081 | 0.4434 | 299.8 s |
| 4 | 1e-3 | 0.4524 | 0.8086 | 0.4431 | 299.2 s |
| 5 | 1e-3 | 0.4519 | 0.8093 | 0.4424 | 301.2 s |
| 6 | 1e-3 | 0.4515 | 0.8093 | 0.4423 | 305.3 s |
| 7 | 1e-3 | 0.4512 | 0.8099 | 0.4417 | 305.8 s |
| 8 | 1e-3 | 0.4509 | 0.8100 | 0.4416 | 303.7 s |
| 9 | 1e-3 | 0.4507 | 0.8101 | 0.4415 | 303.5 s |
| 10 | 1e-3 | 0.4505 | 0.8102 | 0.4415 | 302.3 s |
| 11 | 1e-3 | 0.4503 | 0.8102 | 0.4415 | 307.4 s |
| 12 | 1e-3 | 0.4501 | 0.8103 | 0.4414 | 301.8 s |
| 13 | 1e-3 | 0.4499 | 0.8104 | 0.4413 | 306.0 s |
| 14 | 1e-3 | 0.4498 | 0.8103 | 0.4414 | 307.9 s |
| 15 | 1e-4 | 0.4379 | 0.8132 | 0.4388 | 295.0 s |
| **16 (best)** | **1e-4** | **0.4333** | **0.8134** | **0.4387** | 307.9 s |
| 17 | 1e-4 | 0.4311 | 0.8132 | 0.4390 | 295.2 s |
| 18 | 1e-5 | 0.4265 | 0.8128 | 0.4397 | 306.4 s |

Early stop fired at epoch 18: the best valid AUC was epoch 16 and epochs 17–18
did not improve on it.  The best checkpoint was restored before the test pass.

## Results

| Metric | This run (best epoch 16) | BARS full-rank DCN-V2 | Δ |
| --- | --- | --- | --- |
| test AUC | **0.813739** | **0.814195** | −0.000456 |
| test logloss | **0.438238** | **0.437788** | +0.000450 |

The MLU run is 0.0005 AUC under the published full-rank DCN-V2, which is the
size and sign the paper's own Mix-vs-V2 comparison predicts (Mix −0.0003), so
the backend itself does not move the number.  The paper-protocol run on the
same split (`dcn_mix_paper`, 3 epochs) is 0.0033 AUC lower because it stops
before the LR decay.

## Notes

* Throughput: **108,385 samples/s median** (105,512–111,858 over the 18
  epochs); the rate is flat, i.e. card 4 was not shared.  Each epoch is 8,058
  steps of batch 4096; the final test pass took 3.6 s.
* The valid AUC is flat around 0.8102 for the last six epochs at 1e-3; the 10×
  drop to 1e-4 at epoch 15 gives the +0.003 jump, and the drop to 1e-5 at epoch
  18 ends the run.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; the clock was not sampled while training.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best valid-AUC epoch (16) weights |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads dcn_mix 4
```
