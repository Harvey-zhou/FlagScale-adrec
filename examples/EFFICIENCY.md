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
| DIN / Amazon Electronics | `din_paper_dice` | 32 | 23 (running) | 2,147 | 2,158 | 7 h 46 m so far |
| LightGCN / Gowalla | `lightgcn_paper` | 1024 | 383 (running) | 3,224 | 3,225 | 26 h 44 m so far |
| DIN / Amazon Electronics | `din` (BARS, batch 1024) | 1024 | 4 (running) | **8,492** | 8,493 | 20 min so far |
| DIN / Amazon Electronics | `din_paper_bf16` (BF16) | 32 | 1 | **3,243** | 3,243 | 13 min 24 s |

The DIN paper protocol uses batch 32, so its samples/s is **not** comparable
with the other vendors' batch-1024 column; the `din` BARS row exists precisely
to give a like-for-like number.

## Inference throughput

One warm-up pass + three timed passes, batch 8192 (ads) / 1024 (SASRec).

| Model | Dataset | Rows per pass | FP32 s/pass | FP32 rows/s | BF16 s/pass | BF16 rows/s |
| --- | --- | --- | --- | --- | --- | --- |
| DIN | Amazon Electronics test (384,806) | 384,806 | 1.743 | **220,762** | 2.494 | 154,345 |
| DeepFM | Criteo_x1 test (4,587,167) | 4,587,167 | 4.568 | **1,004,304** | pending | pending |
| SASRec | MovieLens-1M test (6,040 users × 101 candidates) | 6,040 | 0.0675 | **89,455** | 0.200 | 30,488 |
| LightGCN | Gowalla test (29,858 users, full ranking) | pending | | | | |

*Rows per pass* is one test example: an impression for the ads models, a user
with its 101 candidates for SASRec.
 
**BF16 works but is slower on this runtime**: autocast runs without error, yet
DIN loses 30 % and SASRec 66 % throughput.  That is a property of the vendor
kernel path, not of the models.  BF16 training is being measured separately.

## BF16 status

| Item | Status |
| --- | --- |
| Inference autocast | works (`--dtype bf16`), 30–66 % slower |
| Training autocast | works; one full epoch of the 2.6 M rows at batch 32: **3,243 samples/s** (804 s), epoch-1 gAUC 0.8230 vs 0.8245 for FP32 — numerically indistinguishable |
| BF16 vs FP32 training | BF16 looks ~12–18 % faster at batch 32, but the FP32 reference is a 50-epoch mean under heavy contention (1,069–3,314 samples/s spread); a controlled same-load pair has not been run |

BF16 training is therefore usable on this stack, while BF16 *inference* is
slower — the opposite direction, and both are properties of the vendor kernels
rather than of the models.

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
  load.
* DeepFM's inference bench includes reading 4.59 M test rows from the `.npz` on
  the first batch; only one warm-up pass is discarded, so the numbers are steady
  state.
* LightGCN's inference is full-ranking (score matrix on device); the top-k and
  metric are excluded, as the definition requires.
