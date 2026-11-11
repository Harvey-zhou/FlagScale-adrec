# DIN with Dice — Amazon Electronics_x1, paper protocol (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (50 / 50 epochs; best epoch 5) |
| Started / finished | 2026-11-10 12:17:43 → 22:30:14 (CST) |
| Wall clock | 10 h 12 m 31 s (training 36,696 s = 10 h 11 m 36 s) |
| Config | `examples/ads/conf/train/din_paper_dice.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | `d951e7b` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 7, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 4,152,226 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/din_paper_dice-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din_paper_dice 7` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

Identical to the [P800 DIN + Dice run](din_paper_dice_amazonelectronics_x1.md)
and the paper's "DIN with Dice" row (Table 3, 0.8871):

| Setting | Value | Source |
| --- | --- | --- |
| activation | Dice in the activation unit and the final MLP | `[paper]` Table 3, §5.2 |
| Dice epsilon / slope init | 1e-8 / 0 | `[paper]` §5.2, Eq. (9) |
| dataset / split | `amazonelectronics_x1`: vocab item 63,003 + cate 803, train 2,608,764 rows, test 384,806 rows, sequence length ≤ 100 | `[paper]` §6.1 |
| optimizer / schedule | SGD lr 1.0 → 0.1 at step 336,000, no momentum | `[paper]` §6.1 / `[code]` |
| gradient clipping | global norm 5 | `[code]` |
| batch size | 32 | `[paper]` §6.1 |
| embedding | 64 (item) + 64 (cate) | `[code]` |
| attention MLP / DNN | [80, 40] / [80, 40], no softmax | `[code]` / `[paper]` §4.3 |
| dropout / weight decay | 0 / 0 | `[ours]` |
| epochs / evaluation | 50, no early stop, evaluate every epoch; model selection on the test split (the released split has no validation set) | `[code]` |
| metric | gAUC (user-weighted AUC) | `[paper]` §6.3 |
| seed | 1234 | `[code]` |

## Curve

One epoch ≈ 721 s (713.7 s min, 741.5 s max).  Full per-epoch data is in
`history.json`.

| Epoch | LR | Train loss | gAUC | AUC |
| --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.5404 | 0.8262 | 0.8266 |
| 2 | 1.0 | 0.4933 | 0.8329 | 0.8343 |
| 3 | 1.0 | 0.4689 | 0.8439 | 0.8439 |
| 4 | 1.0 | 0.4477 | 0.8552 | 0.8553 |
| **5 (best)** | **1.0** | **0.3978** | **0.8565** | **0.8573** |
| 6 | 0.1 | 0.3728 | 0.8552 | 0.8554 |
| 8 | 0.1 | 0.3454 | 0.8539 | 0.8538 |
| 10 | 0.1 | 0.3167 | 0.8460 | 0.8458 |
| 14 | 0.1 | 0.2594 | 0.8319 | 0.8313 |
| 20 | 0.1 | 0.1869 | 0.8233 | 0.8220 |
| 30 | 0.1 | 0.1138 | 0.8077 | 0.8074 |
| 38 | 0.1 | 0.0837 | 0.8019 | 0.8012 |
| 45 | 0.1 | 0.0674 | 0.8033 | 0.8024 |
| 50 (last) | 0.1 | 0.0595 | 0.7961 | 0.7947 |

The curve has the same shape as the P800 run: gAUC peaks at epoch 5, just before
the learning-rate drop, and then falls monotonically while the training loss
keeps falling — the fixed 50-epoch budget ends well past the overfitting point.
The best checkpoint (epoch 5) was selected by gAUC and restored before the test
pass.

## Results

Final (best epoch 5): test gAUC **0.8565** / AUC 0.8573 / logloss 0.4797.

| Metric | This run | P800 same config | Paper (Table 3) |
| --- | --- | --- | --- |
| test gAUC | 0.8565 | 0.8575 | 0.8871 (paper's TF run 0.8711) |
| test AUC | 0.8573 | 0.8587 | — |
| test logloss | 0.4797 | 0.4816 | — |
| best epoch | 5 | 5 | — |

The MLU run lands 0.0010 gAUC / 0.0014 AUC below the P800 run of the same config
and 0.0306 gAUC below the paper.  The P800 gap to the paper is the known
structure/implementation gap documented in the P800 page, not a backend effect.

## Notes

* **No divergence to NaN.**  On the P800 the same config went NaN at epoch 38;
  here the loss kept falling to 0.0595 at epoch 50 and every evaluation stayed
  finite.  The best checkpoint is epoch 5 either way, so the result is
  unaffected, but the numeric difference is real and stack-dependent (the two
  stacks reduce and clip differently).  Recorded rather than treated as a
  difference in the model.
* Throughput: **3,619 samples/s median** (3,518–3,655 over the 50 epochs); each
  epoch is 81,524 steps of batch 32 over 2,608,764 rows.  The rate is flat with
  no dips, i.e. card 7 was not shared with another job.
* The P800 run took 17 h 13 m of training for the same 50 epochs, so the MLU is
  ~1.7× faster on this model.
* Model selection uses the test split, exactly as the P800 run and the authors'
  released code do; there is no validation set in this dataset release.
* Clock: `cnmon info -q` sampled after the run (card idle) showed 1200 MHz IPU
  vs the 1600 MHz default; the clock was not sampled while training.

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best gAUC epoch (5) weights, 16 MB |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din_paper_dice 7
```
