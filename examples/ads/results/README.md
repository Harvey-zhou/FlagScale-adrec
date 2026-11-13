# Advertising training results

One Markdown file per run, same structure everywhere: run metadata, the settings
that were actually used, the training curve, the final numbers against the
paper/code target, protocol notes, artifacts and the reproduction command.

Protocols (why each value is what it is) live in
[../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md); the underlying raw data is
the `metrics.json` / `history.json` next to each checkpoint.

| Run | Dataset | Status | Result |
| --- | --- | --- | --- |
| [DIN paper protocol](din_paper_amazonelectronics_x1.md) | Amazon Electronics_x1 | finished (50 epochs) | gAUC 0.8555 (paper 0.8818) |
| [DIN + Dice paper protocol](din_paper_dice_amazonelectronics_x1.md) | Amazon Electronics_x1 | finished (50 epochs, NaN after epoch 38) | gAUC 0.8575 (paper 0.8871) |
| [DIN + Dice paper protocol, MLU](din_paper_dice_amazonelectronics_x1_mlu.md) | Amazon Electronics_x1 | finished (50 epochs, best epoch 5) | gAUC 0.8565 (P800 same config 0.8575; paper 0.8871) |
| [DIN, BARS/FuxiCTR, MLU](din_amazonelectronics_x1_mlu.md) | Amazon Electronics_x1 | finished (early stop @10, best epoch 8; FP32 and BF16) | gAUC 0.8821 / AUC 0.8847 (BARS 0.8835 / 0.8860); BF16 0.8824 / 0.8847 |
| [DIN + Dice, released structure, MLU](din_dice_author_amazonelectronics_x1_mlu.md) | Amazon Electronics_x1 | finished (10 epochs, best epoch 3; FP32 and BF16) | gAUC 0.8689 / AUC 0.8714 (P800 same config 0.8658 / 0.8681; paper 0.8871); BF16 0.8690 / 0.8717 |
| [DeepFM, BARS/FuxiCTR, MLU](deepfm_criteo_x1_mlu.md) | Criteo_x1 | finished (early stop @15, best epoch 13) | test AUC 0.8137 / logloss 0.4381 (BARS 0.8137 / 0.4381) |
| [DCN-Mix, BARS/FuxiCTR, MLU](dcn_mix_criteo_x1_mlu.md) | Criteo_x1 | finished (early stop @18, best epoch 16) | test AUC 0.8137 / logloss 0.4382 (BARS full-rank 0.8142 / 0.4378) |
| [DeepFM paper protocol](deepfm_paper_criteo_x1.md) | Criteo_x1 | finished (early stop @6, best epoch 1) | test AUC 0.8040 / logloss 0.4477 (BARS on same split 0.8137 / 0.4381) |
| [DeepFM paper protocol, MLU](deepfm_paper_criteo_x1_mlu.md) | Criteo_x1 | finished (early stop @6, best epoch 1; FP32 and BF16) | test AUC 0.8039 / logloss 0.4477 (P800 same config 0.8040 / 0.4477); BF16 0.8038 / 0.4479 |
| [DCN-Mix paper protocol](dcn_mix_paper_criteo_x1.md) | Criteo_x1 | finished (3 epochs, best epoch 3) | test AUC 0.8104 / logloss 0.4411 (BARS on same split 0.8142 / 0.4378; paper 0.8112 on a different split) |
| [DCN-Mix paper protocol, MLU](dcn_mix_paper_criteo_x1_mlu.md) | Criteo_x1 | finished (3 epochs, best epoch 3) | test AUC 0.8104 / logloss 0.4411 (P800 same config 0.8104 / 0.4411) |

The BARS/FuxiCTR configs (`din.yaml`, `deepfm.yaml`, `dcn_mix.yaml`) are covered
by `EVIDENCE.md`; this directory records the paper-protocol runs and the BARS
full runs that have been reproduced on MLU (marked "BARS/FuxiCTR").
