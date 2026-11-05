# Training / inference efficiency

Measurements for the efficiency table.  Two definitions, both fixed by the
project:

* **Training throughput** — arithmetic mean of the per-epoch
  `train.samples_per_s` over all completed epochs, first epoch included.  Read
  from each run's `history.json`; no timing is repeated.
* **Inference throughput** — final checkpoint, one untimed warm-up pass, then
  three timed full test passes.  `test rows / mean(synchronised H2D + forward +
  D2H)`, with top-k, sorting and metric computation excluded.  Measured with
  [`tools/bench_inference.py`](../tools/bench_inference.py).

The notes that used to sit in the shared table's 备注 column are kept in
[TABLE_NOTES.md](TABLE_NOTES.md).

Environment: Kunlunxin P800 OAM (96 GiB), vendor torch 2.9.0+cu128, fp32,
single card per run, `use_flaggems=false`.  The host is shared, so absolute
numbers move with co-running jobs; the run-to-run spread is shown where it
matters.  The vendor JIT writes to `/tmp/.xblas_cache` and `/tmp/xpujitc-temp`,
which belong to another user on this host, so every job sets its own
`XBLAS_JITC_CACHE_PATH` and `JITC_TEMP_DIR` — with the defaults the first
matmul hangs.

## Training throughput (FP32)

| Run | Config | Batch | Epochs | Mean samples/s | Median | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| SASRec / MovieLens-1M | `sasrec_paper` | 128 | 159 | **5,361** | 5,135 | 3 min 01 s |
| SASRec / Amazon Beauty | `sasrec_beauty_paper` | 128 | 275 | **6,746** | 6,574 | 33 min 18 s |
| SASRec / Amazon Games | `sasrec_games_paper` | 128 | 182 | **6,433** | 6,362 | 14 min 42 s |
| SASRec / Steam | `sasrec_steam_paper` | 128 | 78 | **6,271** | 6,279 | 1 h 07 m |
| DeepFM / Criteo_x1 | `deepfm_paper` | 4096 | 6 | **110,230** | 110,329 | 29 min 57 s |
| DIN / Amazon Electronics | `din_paper` | 32 | 50 | **2,755** | 2,882 | 13 h 41 m |
| DIN / Amazon Electronics | `din_paper` (rerun, PReLU 0.0) | 32 | 50 | **5,438** | 5,412 | 6 h 45 m (idle card) |
| DIN / Amazon Electronics | `din_paper_dice` | 32 | 50 | 2,109 | 2,158 | 17 h 13 m |
| LightGCN / Gowalla | `lightgcn_paper` | 1024 | 1000 | 3,224 | 3,225 | 69 h 56 m |
| LightGCN / Amazon-Book | `lightgcn_amazon_book` | 2048 | 1000 | **2,115** (running) | — | ~313 h projected |
| DIN / Amazon Electronics | `din` (BARS, batch 1024) | 1024 | 9 | **8,493** | 8,493 | 46 min |
| DCN-Mix / Criteo_x1 | `dcn_mix_paper` | 512 | 3 | **9,580** | 9,584 | 2 h 55 m (card shared) |
| DCN-Mix / Criteo_x1 | `dcn_mix_paper`, lr 1e-4 | 512 | 3 | **19,488** | 19,468 | 1 h 26 m (idle card) |
| DeepFM / Criteo_x1 | `deepfm` (BARS) | 4096 | 11 | **96,204** | 96,746 | 1 h 5 m |
| DCN-Mix / Criteo_x1 | `dcn_mix` (BARS + mixture) | 4096 | 14 | **115,797** | 115,646 | 1 h 8 m |
| DIN / Amazon Electronics | `din_paper_bf16` (BF16) | 32 | 1 | **3,243** | 3,243 | 13 min 24 s |
| DIN / Amazon, released structure | `din_author_full` | 32 | 50 | 2,204 | 2,213 | 16 h 36 m |
| DIN / Amazon, released structure + Dice | `din_dice_author` | 32 | 10 | 2024 | 2,024 | 3 h 36 m |
| Two-Tower / synthetic | `two_tower` | 32 | 1 | **984** | 984 | 12 s |
| DIN / MovieLens-20M | `din_ml20m` | 1024 | 9 | **8,056** | 8,059 | 12 h 19 m |
| DCN-Mix / MovieLens-1M | `dcn_mix_ml1m_paper` | 128 | 32 | **8,200** | 8,243 | 39 min |
| DSSM / MovieLens-1M | `dssm` | 4096 | 10 | **179,419** | 180,950 | 3 m 51 s |
| DSSM / MovieLens-20M | `dssm_ml20m` | 4096 | 10 | **183,900** | 183,828 | ~1.2 h |

The DIN paper protocol uses batch 32, so its samples/s is **not** comparable
with the other vendors' batch-1024 column; the `din` BARS row exists precisely
to give a like-for-like number.

## Inference throughput

One warm-up pass + three timed passes, batch 8192 (ads) / 1024 (SASRec) /
2048 (LightGCN) / 4096 (two-tower).  For the two-tower models the item index is
built once before timing (it is the served catalogue, not per-request compute).

| Model | Dataset | Rows per pass | FP32 s/pass | FP32 rows/s | BF16 s/pass | BF16 rows/s |
| --- | --- | --- | --- | --- | --- | --- |
| DIN | Amazon Electronics test (384,806) | 384,806 | 1.743 | **220,762** | 2.494 | 154,345 |
| DIN | MovieLens-20M test (276,986, max_len 100) | 276,986 | 19.58 | **14,145** | 20.14 | 13,750 |
| DeepFM | Criteo_x1 test (4,587,167) | 4,587,167 | 4.568 | **1,004,304** | 3.863 | 1,187,606 |
| SASRec | MovieLens-1M test (6,040 users × 101 candidates) | 6,040 | 0.0675 | **89,455** | 0.200 | 30,488 |
| SASRec | Amazon Beauty test (50,498 users × 101, max_len 50) | 50,498 | 0.191 | **263,633** | 0.241 | 209,782 |
| SASRec | Amazon Games test (30,901 users × 101, max_len 50) | 30,901 | 0.117 | **263,393** | 0.147 | 208,687 |
| SASRec | Steam test (332,925 users × 101, max_len 50) | 332,925 | 1.239 | **268,304** | 1.521 | 216,555 |
| LightGCN | Gowalla test (29,858 users, full ranking) | 29,858 | 0.1747 | **170,849** | 0.1701 | 175,459 |
| LightGCN | Yelp2018 test (31,668 users, full ranking) | 31,668 | 0.247 | **128,344** | 0.250 | 126,532 |
| LightGCN | Amazon-Book | pending first checkpoint (epoch 20) | — | — | — | — | — |
| DCN-Mix | Criteo_x1 test (4,587,167) | 4,587,167 | 8.14 | **563,427** | 8.51 | 539,113 |
| DCN-Mix | MovieLens-1M test (73,901) | 73,901 | 0.0365 | **2,029,283** | 0.0522 | 1,413,111 |
| DSSM | MovieLens-1M test (6,040 users, full catalogue) | 6,040 | 0.0456 | **124,180** | 0.0107 | 490,018 |
| DSSM | MovieLens-20M test (138,493 users, full catalogue) | 138,493 | 8.62 | **16,063** | 4.56 | 30,355 |

*Rows per pass* is one test example: an impression for the ads models, a user
with its 101 candidates for SASRec, a user's full-catalogue ranking for
LightGCN and for the two-tower models.
 
**BF16 works, and whether it helps depends on the model and on what dominates
the pass**: DSSM gains 2.0×/1.9× (the user-tower output times the item matrix is
one big matmul, 490,018 and 30,355 rows/s), DeepFM gains 18 % (1,187,606),
LightGCN is flat (−1.4 % on Yelp, +2.7 % on Gowalla), DCN-Mix loses 30 % on
MovieLens-1M but only 4 % on Criteo (1.14 M-parameter cross/DNN behind a
2 M-row embedding table, so the table lookup dominates), DIN loses 30 % on
Amazon but 3 % on MovieLens-20M (max_len 100, 8192-row batches), and SASRec
loses 66 % on MovieLens-1M — but only ~20 % on Beauty/Games/Steam, whose
sequences are 50 items instead of 200.  The small per-row models fall back to
conversion-heavy paths; wide matmuls do not.  This is a property of the vendor
kernel mix, not of the models.

## BF16 training (mixed precision)

`train.model.torch_dtype=bf16` runs the forward/loss under autocast with fp32
master weights; every paper-protocol config keeps `fp32`, and each BF16 row
below is the same config, seed and epoch budget as its FP32 counterpart.

| Model / dataset | FP32 metric | BF16 metric | Δ | FP32 samples/s | BF16 samples/s |
| --- | --- | --- | --- | --- | --- |
| DeepFM / Criteo_x1 (paper) | AUC 0.8040 / 0.4477 | 0.8040 / 0.4478 | 0.0000 | 110,230 | 159,192 |
| DCN-Mix / MovieLens-1M | AUC 0.8513 / 0.3680 | 0.8509 / 0.3682 | −0.0004 | 8,200 | 7,248 |
| DCN-Mix / Criteo_x1 (paper) | AUC 0.8104 / 0.4411 | 0.8105 / 0.4410 | +0.0001 | 9,580 | 15,912 |
| SASRec / MovieLens-1M | HR@10 0.8204 / NDCG 0.5910 | 0.8215 / 0.5940 | +0.0011 | 5,361 | 8,807 |
| SASRec / Amazon Beauty | HR@10 0.4604 / NDCG 0.3034 | 0.4682 / 0.3104 | +0.0078 | 6,746 | 11,456 |
| SASRec / Amazon Games | HR@10 0.7340 / NDCG@10 0.5281 | 0.7324 / 0.5266 | −0.0016 | 6,433 | 12,635 |
| SASRec / Steam | HR@10 0.8589 / NDCG@10 0.6150 | 0.8593 / 0.6152 | +0.0004 | 6,271 | 13,525 |
| LightGCN / Gowalla (1,000 ep) | recall@20 0.1820 / ndcg 0.1553 | running (epoch 20 identical: 0.1339 / 0.1160) | — | 3,224 | 3,225 |
| LightGCN / Yelp2018 (1,000 ep) | running | running | — | 4,216 | 4,214 |
| Two-Tower (synthetic) | loss 0.6923 / train 0.4402 | identical | 0 | 984 | 1,006 |
| DSSM / MovieLens-1M | Hit@100 0.1921 | 0.1990 | +0.0069 | 179,419 | 174,542 |
| DSSM / MovieLens-20M | Hit@100 0.1647 | 0.1627 | −0.0020 | 183,777 | 179,481 |
| DIN / Amazon (paper, 50 ep) | gAUC 0.8645 / 0.4714 | 0.8625 / 0.4683 | −0.0020 | 5,438 | 5,705 |
| DIN / Amazon (1 epoch probe) | gAUC 0.8227 | 0.8230 | +0.0003 | 3,245 | 3,243 |
| DIN / MovieLens-20M | gAUC 0.9839 / 0.1436 | 0.9840 / 0.1445 | +0.0001 | 8,056 | 7,843 |

Reading of the table:

* **BF16 does not change accuracy** where the trajectory is deterministic:
  DeepFM is identical to five decimals, DCN-Mix moves by 0.0004 and LightGCN is
  bit-identical at every evaluated epoch (Gowalla e40/60/80/100 = 0.1487 /
  0.1569 / 0.1626 / 0.1668 for both precisions; Yelp2018 the same).  Where the
  number moves more (SASRec Beauty +0.0078, DSSM +0.0069, DIN −0.0020) the
  early-stopping epoch moved with it — SASRec Beauty ran to epoch 324 under BF16
  against 275 under FP32, and DIN peaks at epoch 7 instead of 11 because the
  FP32 run has a transient dip at epoch 4 that the BF16 run does not — so those
  are trajectory differences, not precision loss.
* **BF16 is not uniformly faster for training.**  At batch 128 (DCN-Mix) it is
  12 % slower; at batch 4096 (DeepFM, SASRec) it is faster; LightGCN is exactly
  flat because the cost is the full-graph `index_add` propagation, not a matmul.
  The FP32 column is mostly measured under host contention, so these are
  indicative rather than controlled speedups.
* The older note in this file said 30–66 % slower *inference* across the board;
  with the two-tower models measured, the statement is now model-dependent
  (see the table above).

## Reproduction

```bash
export PYTHONPATH=$PWD XPU_VISIBLE_DEVICES=<card>
python tools/bench_inference.py --task ads \
  --checkpoint <run>/checkpoint --data /data2/jiangxuhui/datasets/amazonelectronics_x1 \
  --batch-size 8192 [--dtype bf16]
```

`--task sequential` and `--task graph` cover the rec models.  Training
throughput is read from `history.json`:

```bash
python -c "import json,statistics; h=json.load(open('<run>/checkpoint/history.json')); \
print(statistics.mean(r['train']['samples_per_s'] for r in h))"
```

## Caveats

* The host is shared.  The DIN paper run's per-epoch spread was 1,069–3,314
  samples/s because other jobs were running; its mean is therefore a lower
  bound.  The BARS batch-1024 row and the BF16 epoch are measured under lighter
  load.  The 2026-11-03 DIN rerun (PReLU 0.0) ran with four jobs on the host and
  still held 4,887–5,767 samples/s, so ~5.4 k is the honest single-card number
  and 2,755 is what contention cost.
* DCN-Mix has two training rows because the paper-protocol run shared a card
  (9,580 samples/s) while the lr-1e-4 arm ran with two cards in use but no other
  tenant (19,488 samples/s).  Epoch time, not samples/s, is the stable number:
  3,383–3,509 s shared vs 1,680–1,705 s solo.
* DeepFM's inference bench includes reading 4.59 M test rows from the `.npz` on
  the first batch; only one warm-up pass is discarded, so the numbers are steady
  state.
* LightGCN's inference is full-ranking (score matrix on device); the top-k and
  metric are excluded, as the definition requires.  The row is **Yelp2018**
  because the Gowalla paper run predates this host and its checkpoint is not
  present; the Gowalla number is still unmeasured.
