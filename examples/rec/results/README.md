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
| [LightGCN paper protocol](lightgcn_paper_gowalla.md) | Gowalla | running | pending |

The reference-code configs (`sasrec.yaml`, `lightgcn.yaml`, …) are covered by
`EVIDENCE.md`; this directory records the paper-protocol runs.
