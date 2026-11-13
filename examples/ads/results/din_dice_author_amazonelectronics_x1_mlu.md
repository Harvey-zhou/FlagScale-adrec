# DIN with Dice, released model structure — Amazon Electronics_x1 (Cambricon MLU)

| Field | Value |
| --- | --- |
| Status | **finished** (10 / 10 epochs; best epoch 3) |
| Started / finished | 2026-11-11 16:57:43 → 19:02:08 (CST) |
| Wall clock | 2 h 04 m 25 s (training 7,412.9 s = 2 h 03 m 33 s) |
| Config | `examples/ads/conf/train/din_dice_author.yaml` |
| Protocol document | [../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md) |
| Code commit | `cda3072` (working tree clean) |
| Device | Cambricon MLU590-M9, node B (MLU driver 6.2.29), card 5, fp32 |
| Stack | torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3, CNNL 2.2.18, CNCL 1.30.10, MLU-OPS 1.13.0 |
| Parameters | 4,179,882 |
| Output | `/home/zhouhao/flagos-adrec-cambricon/out/ads/ckpts/din_dice_author-mlu/checkpoint/` |
| Command | `bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din_dice_author 5` |

Strict mode: `ENABLE_FALLBACK_TO_CPU=0`; the launcher's log scan found no
operator falling back to the CPU and no operator missing an MLU kernel.

## Settings

Same protocol as the paper-structure Dice run
([`din_paper_dice_amazonelectronics_x1_mlu.md`](din_paper_dice_amazonelectronics_x1_mlu.md)),
with the pieces that only the authors' released `model_dice.py` has switched on:
sigmoid in the attention MLP, Dice only in the DNN, softmax after `sqrt(d)`,
`hist_bn`/`hist_fcn`, the `[u, i, u*i]` MLP input and its BatchNorm.

| Setting | Value | Source |
| --- | --- | --- |
| activation | Dice in the final MLP, sigmoid in the attention MLP | `[paper]` Table 3; `[code]` |
| Dice epsilon / slope init | 1e-8 / 0 | `[paper]` §5.2, Eq. (9) |
| dataset / split | `amazonelectronics_x1`: train 2,608,764 rows, test 384,806 rows, sequence length ≤ 100 | `[paper]` §6.1 |
| optimizer / schedule | SGD lr 1.0 → 0.1 at step 336,000, no momentum | `[paper]` §6.1 / `[code]` |
| gradient clipping | global norm 5 | `[code]` |
| batch size | 32 | `[paper]` §6.1 |
| attention MLP / DNN | [80, 40] / [80, 40] | `[code]` |
| dropout / weight decay | 0 / 0 | `[ours]` |
| epochs / evaluation | 10, no early stop, evaluate every epoch; model selection on the test split (the released split has no validation set) | `[ours]` |
| metric | gAUC (user-weighted AUC) | `[paper]` §6.3 |
| seed | 1234 | `[code]` |

Ten epochs is the ablation length used for the P800 run of this config: the
released structure peaks at epoch 2–3 and then overfits.

## Curve

One epoch ≈ 741 s (701–773 s).  Full per-epoch data is in `history.json`.

| Epoch | LR | Train loss | gAUC | AUC |
| --- | --- | --- | --- | --- |
| 1 | 1.0 | 0.4996 | 0.858760 | 0.860686 |
| 2 | 1.0 | 0.4300 | 0.864997 | 0.867262 |
| **3 (best)** | **1.0** | **0.3812** | **0.868916** | **0.871356** |
| 4 | 1.0 | 0.3270 | 0.863583 | 0.864854 |
| 5 | 1.0 | 0.2324 | 0.855584 | 0.857734 |
| 6 | 0.1 | 0.1945 | 0.853147 | 0.854865 |
| 7 | 0.1 | 0.1751 | 0.849841 | 0.852169 |
| 8 | 0.1 | 0.1594 | 0.847149 | 0.849189 |
| 9 | 0.1 | 0.1457 | 0.842861 | 0.844603 |
| 10 (last) | 0.1 | 0.1342 | 0.842211 | 0.843374 |

The curve peaks at epoch 3 and then falls while the training loss keeps
falling — the same shape as the P800 run, just shifted by one epoch.  The best
checkpoint (epoch 3, chosen by gAUC) was restored before the test pass.

## Results

Final (best epoch 3): test gAUC **0.868916** / AUC 0.871356 / logloss 0.460640.

| Metric | This run | P800 same config | Paper (Table 3) | Authors' README |
| --- | --- | --- | --- | --- |
| test gAUC | **0.868916** | 0.8658 | 0.8871 | 0.8711 |
| test AUC | **0.871356** | 0.8681 | — | — |
| test logloss | 0.460640 | — | — | — |
| best epoch | 3 | 2 | — | — |

The MLU run lands 0.0031 gAUC above the P800 run of the same config, 0.0022
below the authors' own 0.8711, and 0.0182 below the paper's 0.8871 — the same
implementation gap the paper structure shows on this data.

## BF16

Full run of the same config with `train.model.torch_dtype=bf16`
(`2026-11-13 10:38:55 → 12:52:50`, card 1, 10 epochs, best epoch 3 — the same
schedule as FP32).  Autocast: weights and optimizer state stay fp32, and the
run wrote to a separate `ADREC_OUT_ROOT` (`out-bf16`).

| Metric | FP32 (best epoch 3) | BF16 (best epoch 3) | Δ |
| --- | --- | --- | --- |
| test gAUC | 0.868916 | 0.869033 | +0.000117 |
| test AUC | 0.871356 | 0.871681 | +0.000325 |
| test logloss | 0.460640 | 0.459961 | −0.000679 |
| epochs run | 10 | 10 | |
| training time | 7,412.9 s | 7,953.2 s | +7.3 % |

Both curves peak at epoch 3 and decay the same way; the per-epoch gAUC agrees
to ≤ 0.004 (largest at epoch 2, where BF16 is ahead), and the late-epoch
logloss inflation (0.61 → 0.92) appears in both.  Final deltas are ≤ 6.8e-4,
so BF16 is accuracy-safe for this config.

Throughput: **3,334 samples/s mean** (median 3,338) vs 3,584 / 3,632 FP32, i.e.
**7.0 % slower** per epoch (779–793 s vs 701–773 s).  This is one of the
small-batch (32) DIN configs where the conversion overhead outweighs the
matmul gain, the same direction as the DeepFM / DCN-Mix BARS pairs and the
opposite of the batch-1024 BARS `din` (+17.3 %).

## Notes

* **No NaN.**  On the P800 the same config diverged at epoch 3 (best checkpoint
  epoch 2); here epoch 3 is the best epoch and the loss stayed finite for all
  ten epochs (0.1342 at the end).  The best-checkpoint logic makes the P800
  divergence harmless, and the MLU run simply does not show it — the two stacks
  reduce and clip differently under constant SGD 0.1.
* Throughput: **3,632 samples/s median** (3,376–3,721 over the 10 epochs,
  mean 3,584); each epoch is 81,524 steps of batch 32 over 2,608,764 rows.
* The P800 run took 3 h 36 m of training for the same 10 epochs (2,024
  samples/s), so node B is ~1.8× faster on this config.
* Model selection uses the test split, exactly as the P800 run and the authors'
  released code do; there is no validation set in this dataset release.
* Clock: not sampled while training; idle cards report 1200 MHz with
  `Clock Setting: Active`, and under load the clock reaches 1600 MHz (see
  `CAMBRICON.md`).

## Artifacts

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | best gAUC epoch (3) weights |
| `checkpoint/metrics.json` | full summary + per-epoch history |
| `checkpoint/history.json` | per-epoch history, written incrementally |
| `checkpoint/config.json` | model config + dataset meta |

## Reproduce

```bash
export ADREC_DATA_ROOT=/home/zhouhao/flagos-adrec-cambricon/data
export ADREC_OUT_ROOT=/home/zhouhao/flagos-adrec-cambricon/out
export CAMBRICON_IMAGE=cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312

bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din_dice_author 5
```
