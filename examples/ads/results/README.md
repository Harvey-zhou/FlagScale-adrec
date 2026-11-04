# Advertising training results

One Markdown file per run, same structure everywhere: run metadata, the settings
that were actually used, the training curve, the final numbers against the
paper/code target, protocol notes, artifacts and the reproduction command.

Protocols (why each value is what it is) live in
[../DIN_PAPER_PROTOCOL.md](../DIN_PAPER_PROTOCOL.md); the underlying raw data is
the `metrics.json` / `history.json` next to each checkpoint.

| Run | Dataset | Status | Result |
| --- | --- | --- | --- |
| [DIN paper protocol](din_paper_amazonelectronics_x1.md) | Amazon Electronics_x1 | finished (50 epochs, best epoch 11) | gAUC **0.8645** (paper 0.8818, authors' README 0.8698; the previous run with PReLU init 0.25 gave 0.8555) |
| [DIN + Dice paper protocol](din_paper_dice_amazonelectronics_x1.md) | Amazon Electronics_x1 | finished (50 epochs, NaN after epoch 38; reproduced twice) | gAUC 0.8575 (paper 0.8871) — 0.0070 **below** the PReLU rerun's 0.8645 |
| [DeepFM paper protocol](deepfm_paper_criteo_x1.md) | Criteo_x1 | finished (early stop @6, best epoch 1) | test AUC 0.8040 / logloss 0.4477 (BARS on this host 0.8112 / 0.4406) |
| [DCN-Mix paper protocol](dcn_mix_ml1m_paper_movielens_1m.md) | MovieLens-1M | finished (32 epochs, best epoch 7) | test AUC **0.8513** / logloss 0.3680 (paper 0.8964 / 0.3160) — a 6-arm ladder tops out at 0.8540, so the remaining 0.042 is not lr, loss or model size |
| [DCN-Mix paper protocol](dcn_mix_paper_criteo_x1.md) | Criteo_x1 | finished (3 epochs, best epoch 3) | test AUC 0.8104 / logloss 0.4411 (BARS on this host 0.8139 / 0.4380; paper 0.8112 on a different split) |
| [DCN-Mix lr ablation](dcn_mix_paper_lr_ablation_criteo_x1.md) | Criteo_x1 | finished (3 arms × 3 epochs) | 1e-4: AUC 0.8115 / 0.4401; 5e-4: 0.8104 / 0.4411; 1e-3: 0.8075 / 0.4438 — lower is better, the BARS-rate hypothesis is falsified |
| [BARS reference configs](bars_baseline_criteo_x1.md) | Criteo_x1 | finished (early stopping) | DeepFM 0.8112 / 0.4406 (BARS 0.8137), DCN-Mix 0.8139 / 0.4380 (BARS 0.8142) — the baseline is now measured on this host |
| [DeepFM lr ablation](deepfm_paper_lr_ablation_criteo_x1.md) | Criteo_x1 | finished (3 arms) | 1e-3: 0.8040; 3e-4: 0.8055; 1e-4: 0.8037 — the step size is not the explanation, every arm peaks in 1–3 epochs |
| [DeepFM regularisation decomposition](deepfm_paper_regularisation_criteo_x1.md) | Criteo_x1 | finished (3 arms) | +BatchNorm: 0.8045 (no help); +embedding L2 1e-5: **0.8102**; +L2+BN+dropout 0.1: **0.8111** — matches BARS 0.8112 to 0.0001, so the gap is entirely regularisation |

The BARS/FuxiCTR configs (`din.yaml`, `deepfm.yaml`, `dcn_mix.yaml`) are covered
by `EVIDENCE.md`; this directory records the paper-protocol runs, their
ablations, and the BARS baselines re-run on this host
([bars_baseline_criteo_x1.md](bars_baseline_criteo_x1.md)).
