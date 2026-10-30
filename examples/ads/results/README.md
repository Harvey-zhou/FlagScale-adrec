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
| [DIN + Dice paper protocol](din_paper_dice_amazonelectronics_x1.md) | Amazon Electronics_x1 | running | target 0.8871 |
| [DeepFM paper protocol](deepfm_paper_criteo_x1.md) | Criteo_x1 | finished (early stop @6, best epoch 1) | test AUC 0.8040 / logloss 0.4477 (BARS on same split 0.8137 / 0.4381) |

The BARS/FuxiCTR configs (`din.yaml`, `deepfm.yaml`, `dcn_mix.yaml`) are covered
by `EVIDENCE.md`; this directory records the paper-protocol runs.
