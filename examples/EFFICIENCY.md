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

Environment: Kunlunxin P800 OAM (96 GiB), vendor torch 2.9.0+cu128, fp32,
single card per run, `use_flaggems=false`.  The host is shared, so absolute
numbers move with co-running jobs; the run-to-run spread is shown where it
matters.

## Training throughput (FP32)

| Run | Config | Batch | Epochs | Mean samples/s | Median | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| SASRec / MovieLens-1M | `sasrec_paper` | 128 | 159 | **5,361** | 5,135 | 3 min 01 s |
| DeepFM / Criteo_x1 | `deepfm_paper` | 4096 | 6 | **110,230** | 110,329 | 29 min 57 s |
| DIN / Amazon Electronics | `din_paper` | 32 | 50 | **2,755** | 2,882 | 13 h 41 m |
| DIN / Amazon Electronics | `din_paper_dice` | 32 | 50 | 2,109 | 2,158 | 17 h 13 m |
| LightGCN / Gowalla | `lightgcn_paper` | 1024 | 1000 | 3,224 | 3,225 | 69 h 56 m |
| DIN / Amazon Electronics | `din` (BARS, batch 1024) | 1024 | 9 | **8,493** | 8,493 | 46 min |
| DCN-Mix / Criteo_x1 | `dcn_mix_paper` | 512 | blocked | | | vendor JIT (bmm) failed; fixed to 2-D matmuls, waiting for a free card |
| DIN / Amazon Electronics | `din_paper_bf16` (BF16) | 32 | 1 | **3,243** | 3,243 | 13 min 24 s |
| DIN / Amazon, released structure | `din_author_full` | 32 | 50 | 2,204 | 2,213 | 16 h 36 m |
| DIN / Amazon, released structure + Dice | `din_dice_author` | 32 | 10 | 2024 | 2,024 | 3 h 36 m |

The DIN paper protocol uses batch 32, so its samples/s is **not** comparable
with the other vendors' batch-1024 column; the `din` BARS row exists precisely
to give a like-for-like number.

## Inference throughput

One warm-up pass + three timed passes, batch 8192 (ads) / 1024 (SASRec).

| Model | Dataset | Rows per pass | FP32 s/pass | FP32 rows/s | BF16 s/pass | BF16 rows/s |
| --- | --- | --- | --- | --- | --- | --- |
| DIN | Amazon Electronics test (384,806) | 384,806 | 1.743 | **220,762** | 2.494 | 154,345 |
| DeepFM | Criteo_x1 test (4,587,167) | 4,587,167 | 4.568 | **1,004,304** | 3.863 | 1,187,606 |
| SASRec | MovieLens-1M test (6,040 users × 101 candidates) | 6,040 | 0.0675 | **89,455** | 0.200 | 30,488 |
| LightGCN | Gowalla test (29,858 users, full ranking) | pending | | | | |
| DSSM | MovieLens-1M test (6,040 users, full-catalogue retrieval) | pending | | | | |

*Rows per pass* is one test example: an impression for the ads models, a user
with its 101 candidates for SASRec.
 
**BF16 works, and whether it helps depends on the model**: autocast runs
without error; DeepFM gains 18 % (1,187,606 rows/s), while DIN loses 30 % and
SASRec loses 66 %.  The small batch-32 / 1-head models fall back to
conversion-heavy paths, the large embedding+MLP model does not.  This is a
property of the vendor kernel mix, not of the models.

### Cambricon MLU590-M9 (2026-11-12)

Same definition, `--device mlu:0`, batch 1024 for SASRec / LightGCN and 8192 for
DSSM / ads (aligned with the tables above; medians of the timed passes):

| Model | Dataset | Rows per pass | FP32 s/pass | FP32 rows/s | BF16 s/pass | BF16 rows/s |
| --- | --- | --- | --- | --- | --- | --- |
| SASRec | MovieLens-1M test (6,040 users × 101 candidates) | 6,040 | 0.0298 | 202,731 | 0.0259–0.0516 | 117,134–225,274 (unstable) |
| SASRec | Games test (30,901 users) | 30,901 | 0.1024 | 301,835 | 0.1169 | 264,298 |
| SASRec | Beauty test (50,498 users) | 50,498 | 0.1645 | 307,047 | 0.1857 | 271,996 |
| SASRec | Steam test (332,925 users) | 332,925 | 1.0905 | 305,305 | 1.2066 | 275,929 |
| DSSM | MovieLens-1M test (full-catalogue retrieval) | 6,040 | 0.0062 | 975,888 | 0.0065 | 924,686 |
| DIN | AmazonElectronics test (384,806 rows) | 384,806 | 0.5738 | 670,636 | 0.5825 | 660,652 |
| LightGCN | Gowalla test (full ranking) | 29,858 | 0.0121 | 2,457,798 | 0.0128 | 2,260,501 |
| LightGCN | Yelp2018 test (full ranking) | 31,668 | 0.0152 | 2,086,298 | 0.0149 | 2,118,588 |

`autocast_active=true` for every BF16 pass.  DSSM online-only (catalogue
encoding excluded): FP32 1,432,879 / BF16 1,368,100 rows/s (medians).

On MLU, BF16 inference loses on six of the eight rows (5–12 %); DIN is neutral
(−1.5 %) and Yelp2018 (+1.5 %) is within noise.  The MovieLens-1M row is the
exception that does not hold up: four runs gave BF16 medians of 117,134 /
217,161 / 225,274 rows/s against a stable FP32 192k–211k, so it is recorded as
a range rather than a speedup.  Unlike the P800 SASRec row (always slower under
BF16), the MLU result is mixed, i.e. the vendor kernel mix — not the model shape
— decides.  The matching MLU training results (SASRec 8–15 % slower per epoch,
DSSM flat with a lower metric, LightGCN unchanged) and the exact commands are in
[`CAMBRICON.md`](CAMBRICON.md#bf16-batch-results-2026-11-1112).

## BF16 status (P800)

| Item | Status |
| --- | --- |
| Inference autocast | works (`--dtype bf16`), 30–66 % slower |
| Training autocast | works; one full epoch of the 2.6 M rows at batch 32: **3,243 samples/s** (804 s), epoch-1 gAUC 0.8230 vs 0.8245 for FP32 — numerically indistinguishable |
| BF16 vs FP32 training | BF16 looks ~12–18 % faster at batch 32, but the FP32 reference is a 50-epoch mean under heavy contention (1,069–3,314 samples/s spread); a controlled same-load pair has not been run |

BF16 training is therefore usable on this stack, while BF16 *inference* is
slower — the opposite direction, and both are properties of the vendor kernels
rather than of the models.  On MLU the sign is model-dependent; see the MLU
table above.

## Reproduction

```bash
export PYTHONPATH=$PWD XPU_VISIBLE_DEVICES=<card>
python tools/bench_inference.py --task ads \
  --checkpoint <run>/checkpoint --data /data2/jiangxuhui/datasets/amazonelectronics_x1 \
  --batch-size 8192 [--dtype bf16]
```

`--task sequential`, `--task graph` and `--task two_tower` cover the rec models.

`--task two_tower` (DSSM) is full-catalogue retrieval and reports two numbers,
both in test queries per second, over `--repeat` passes (default 10, because a
pass on MovieLens-1M is short; read the median):

* `rows_per_s_*` — the whole pass: item tower over the full catalogue, user
  tower over the test queries, and the query x catalogue score matrix;
* `online_rows_per_s_*` — the same pass without the catalogue encoding, which a
  serving system builds offline.

As for LightGCN, top-k and Hit/NDCG are excluded.  `--task synthetic_two_tower`
times TorchRec's random-data two-tower example as pair scoring (no `--data`).
Training throughput is read from `history.json`:

```bash
python -c "import json,statistics; h=json.load(open('<run>/checkpoint/history.json')); \
print(statistics.mean(r['train']['samples_per_s'] for r in h))"
```

## Caveats

* The host is shared.  The DIN paper run's per-epoch spread was 1,069–3,314
  samples/s because other jobs were running; its mean is therefore a lower
  bound.  The BARS batch-1024 row and the BF16 epoch are measured under lighter
  load.
* DeepFM's inference bench includes reading 4.59 M test rows from the `.npz` on
  the first batch; only one warm-up pass is discarded, so the numbers are steady
  state.
* LightGCN's inference is full-ranking (score matrix on device); the top-k and
  metric are excluded, as the definition requires.
