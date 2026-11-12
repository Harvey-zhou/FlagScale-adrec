# Recommendation training results

One Markdown file per run, same structure everywhere: run metadata, the settings
that were actually used, the training curve, the final numbers against the
paper/code target, protocol notes, artifacts and the reproduction command.

Protocols (why each value is what it is) live in
[../PAPER_PROTOCOLS.md](../PAPER_PROTOCOLS.md); the underlying raw data is the
`metrics.json` / `history.json` next to each checkpoint.

| Run | Dataset | Status | Result |
| --- | --- | --- | --- |
| [SASRec paper protocol](sasrec_paper_movielens_1m.md) | MovieLens-1M | finished (early stop @159) | HR@10 0.8204 / NDCG@10 0.5910 (paper 0.8245 / 0.5905) |
| [SASRec paper protocol](sasrec_paper_amazon_beauty.md) | Amazon Beauty | finished (early stop @275) | HR@10 0.4604 / NDCG@10 0.3034 (paper 0.4854 / 0.3219) |
| [SASRec paper protocol](sasrec_paper_amazon_games.md) | Amazon Games | finished (early stop @182) | HR@10 0.7340 / NDCG@10 0.5281 (paper 0.7410 / 0.5360) |
| [SASRec paper protocol](sasrec_paper_steam.md) | Steam | finished (early stop @78) | HR@10 0.8589 / NDCG@10 0.6150 (paper 0.8729 / 0.6306) |
| [LightGCN paper protocol](lightgcn_paper_gowalla.md) | Gowalla | finished (1,000 epochs) | recall@20 0.1820 / ndcg@20 0.1553 (paper 0.1823 / 0.1555) |
| [LightGCN paper protocol, batch 2048](lightgcn_paper_yelp2018_b2048.md) | Yelp2018 | finished (1,000 epochs, FP32 + BF16) | recall@20 **0.0576** / ndcg@20 0.0465 (paper 0.0639 / 0.0525) — 604 k optimizer steps instead of the paper's 1.21 M; BF16 and FP32 identical at all 50 evaluation points |
| LightGCN paper protocol, batch 1024 | Yelp2018 | running (paper setting, `*-b1024-*-v2`) | FP32 at 360/1000, BF16 at 499/1000 (2026-11-12); the arm that will test whether the shortfall is the step budget |
| LightGCN | Amazon-Book | **interrupted** by the same fault at epoch 143; restarted 2026-11-09 as `amazon-book-fp32-v2` (in progress) | partial, best 0.0383/0.0297 (e140); paper 0.0410/0.0318 |
| [Two-Tower (TorchRec)](two_tower_synthetic_torchrec.md) | synthetic (random) | finished (100 iterations) | no accuracy target; train loss 0.4402 / held-out 0.6923 (≈ ln 2), identical to the CPU reference |
| [DSSM (torch-rechub)](dssm_movielens_1m.md) | MovieLens-1M | finished (10 epochs) | Hit@100 **0.1921** / NDCG@100 0.0430 (torch-rechub README 0.1474; matches the 2026-10-15 vendor run to 4 decimals) |
| [DSSM (torch-rechub)](dssm_movielens_20m.md) | MovieLens-20M | finished (10 epochs) | Hit@100 **0.1647** / NDCG@100 0.0393 (best epoch 9: 0.1665); no published reference; user tower is user_id + history because ML-20M has no demographics |

The reference-code configs (`sasrec.yaml`, `lightgcn.yaml`, …) are covered by
`EVIDENCE.md`; this directory records the paper-protocol runs.
