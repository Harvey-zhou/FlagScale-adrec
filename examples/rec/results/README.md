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
| [SASRec paper protocol, MLU](sasrec_paper_movielens_1m_mlu.md) | MovieLens-1M | finished (early stop @163, best 143) | HR@10 0.8174 / NDCG@10 0.5895 (paper 0.8245 / 0.5905) |
| [DSSM, MLU](dssm_movielens_1m_mlu.md) | MovieLens-1M | finished (10 epochs) | Hit@100 0.1599 / NDCG@100 0.0334 (torch-rechub Hit@100 14.74%) |
| [SASRec paper protocol](sasrec_paper_amazon_beauty.md) | Amazon Beauty | finished (early stop @275) | HR@10 0.4604 / NDCG@10 0.3034 (paper 0.4854 / 0.3219) |
| [SASRec paper protocol](sasrec_paper_amazon_games.md) | Amazon Games | finished (early stop @182) | HR@10 0.7340 / NDCG@10 0.5281 (paper 0.7410 / 0.5360) |
| [SASRec paper protocol](sasrec_paper_steam.md) | Steam | finished (early stop @78) | HR@10 0.8589 / NDCG@10 0.6150 (paper 0.8729 / 0.6306) |
| [LightGCN paper protocol](lightgcn_paper_gowalla.md) | Gowalla | finished (1,000 epochs) | recall@20 0.1820 / ndcg@20 0.1553 (paper 0.1823 / 0.1555) |
| [LightGCN code settings, MLU](lightgcn_gowalla_mlu.md) | Gowalla | finished (1,000 epochs, 2-card) | recall@20 0.1818 / ndcg@20 0.1553 (released code 0.1824 / 0.1547) |
| LightGCN paper protocol | Yelp2018 | P800 run did not finish | code settings reproduced on MLU: 0.0632 / 0.0522 (see below) |

The reference-code configs (`sasrec.yaml`, `lightgcn.yaml`, …) are covered by
`EVIDENCE.md`; the code-settings LightGCN Gowalla run has its own page above
(`lightgcn_gowalla_mlu.md`, 2-card), and the code-settings LightGCN Yelp2018 run
is `lightgcn_yelp2018_code.yaml` (the P800 run above never finished; the MLU run
reproduced the code reference at 0.0632 / 0.0522 — see `../../CAMBRICON.md`).
