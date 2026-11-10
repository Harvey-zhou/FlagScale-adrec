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
| [DeepFM paper protocol](deepfm_paper_criteo_x1.md) | Criteo_x1 | finished (early stop @6, best epoch 1) | test AUC 0.8040 / logloss 0.4477 (BARS on same split 0.8137 / 0.4381) |
| [DeepFM paper protocol, MLU](deepfm_paper_criteo_x1_mlu.md) | Criteo_x1 | finished (early stop @6, best epoch 1) | test AUC 0.8039 / logloss 0.4477 (P800 same config 0.8040 / 0.4477) |
| [DCN-Mix paper protocol](dcn_mix_paper_criteo_x1.md) | Criteo_x1 | finished (3 epochs, best epoch 3) | test AUC 0.8104 / logloss 0.4411 (BARS on same split 0.8142 / 0.4378; paper 0.8112 on a different split) |
| [DCN-Mix paper protocol, MLU](dcn_mix_paper_criteo_x1_mlu.md) | Criteo_x1 | finished (3 epochs, best epoch 3) | test AUC 0.8104 / logloss 0.4411 (P800 same config 0.8104 / 0.4411) |

The BARS/FuxiCTR configs (`din.yaml`, `deepfm.yaml`, `dcn_mix.yaml`) are covered
by `EVIDENCE.md`; this directory records the paper-protocol runs.
