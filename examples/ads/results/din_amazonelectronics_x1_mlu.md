# DIN — Amazon Electronics_x1, BARS/FuxiCTR protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (early stopped at epoch 10; best epoch 8) |
| Started / finished | 2026-11-11 10:26:41 → 10:37:16 (CST) |
| Wall clock | 10 min 35 s (training 571.7 s) |
| Config | `examples/ads/conf/train/din.yaml` |
| Code commit | `08fa8dd` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 1, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 5,403,522 (identical shape to the BARS model) |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/din-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din 1` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

This is the BARS/FuxiCTR reference config
(`ranking/ctr/DIN/DIN_amazonelectronics_x1`, FuxiCTR v2.0.1), not the paper
protocol of [`din_paper_dice_amazonelectronics_x1_mlu.md`](din_paper_dice_amazonelectronics_x1_mlu.md).

| Setting | Value | Source |
| --- | --- | --- |
| dataset / split | `amazonelectronics_x1`: item vocab 63,003 + cate 803, train 2,608,764 rows, test 384,806 rows, max length 100 | BARS |
| embedding | 64 (item) + 64 (cate) | BARS |
| attention unit | [512, 256], Dice, dropout 0.1, **softmax** over the candidate set | BARS |
| DNN | [1024, 512, 256], Dice, BatchNorm, dropout 0.1 | BARS |
| optimizer | Adam, lr 5e-4 (observed drops to 5e-5 at epoch 6 and 5e-6 at epoch 10) | BARS |
| embedding regularizer | 5e-3 | BARS |
| gradient clipping | global norm 10 | BARS |
| batch size | 1024 | BARS |
| epochs / early stop | 100 cap, patience 2 on gAUC, model selection on the **test** split | BARS |
| seed | 20222023 | BARS |

The test split doubles as the selection split, exactly as BARS runs it, so this
row is directly comparable to the published number.  It is not comparable to the
paper-protocol DIN runs, which use SGD lr 1.0, batch 32 and no embedding
regularizer.

## Curve

2,548 steps per epoch, ~42.9 s.

| Epoch | LR | Train loss | gAUC | AUC | Logloss |
| --- | --- | --- | --- | --- | --- |
| 1 | 5e-4 | 0.6332 | 0.8394 | 0.8412 | 0.4903 |
| 2 | 5e-4 | 0.5848 | 0.8477 | 0.8495 | 0.4798 |
| 3 | 5e-4 | 0.5687 | 0.8541 | 0.8559 | 0.4702 |
| 4 | 5e-4 | 0.5613 | 0.8575 | 0.8590 | 0.4650 |
| 5 | 5e-4 | 0.5566 | 0.8559 | 0.8585 | 0.4665 |
| 6 | 5e-5 | 0.4660 | 0.8764 | 0.8794 | 0.4367 |
| **8 (best)** | **5e-5** | **0.4064** | **0.8821** | **0.8847** | **0.4322** |
| 9 | 5e-5 | 0.3939 | 0.8813 | 0.8838 | 0.4389 |
| 10 | 5e-6 | 0.3335 | 0.8793 | 0.8814 | 0.4738 |

Early stop fired at epoch 10: the best gAUC was epoch 8 and epochs 9–10 did not
improve on it.  The best checkpoint was restored before the test pass.

## Results

| Metric | This run (best epoch 8) | BARS published | Δ |
| --- | --- | --- | --- |
| test gAUC | **0.8821** | **0.8835** | −0.0014 |
| test AUC | 0.8847 | 0.8860 | −0.0013 |
| test logloss | 0.4322 | 0.4302 | +0.0020 |

This reproduces the BARS reference to 0.0014 gAUC / 0.0013 AUC, i.e. the MLU
backend does not change the result.  It also confirms that the ~0.03 gAUC gap
seen in the paper-protocol runs is a protocol/implementation gap, not a backend
one: the same cards give 0.8565 for `din_paper_dice` and 0.8821 here.

## Notes

* Throughput: **60,828 samples/s median** (60,567–60,885 over the ten epochs),
  i.e. 42.9 s per epoch of 2,548 steps at batch 1024.  The final test pass took
  12.7 s.
* The run took the card for itself; card 1 showed no other process and the
  samples/s stayed within 0.5 % across epochs.
* BARS selects on the test split; the same is done here for comparability.  The
  `din_paper` runs on node A use the paper protocol and are not comparable
  metric-for-metric.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; the clock was not sampled while training.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best gAUC epoch (8) weights |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din 1
```
