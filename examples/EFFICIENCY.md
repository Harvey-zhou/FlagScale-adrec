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

### Cambricon MLU590-M9 (2026-11-11)

Node B (driver 6.2.29), fp32 unless marked BF16, single card per run; the
2-card LightGCN row uses two cards.  Training throughput is the same
`history.json` mean as above:

| Run | Config | Batch | Epochs | Mean samples/s | Median | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| DIN / Amazon Electronics | `din` (BARS) | 1024 | 10 | **60,805** | 60,828 | 9 min 32 s |
| DIN / Amazon, released + Dice | `din_dice_author` | 32 | 10 | **3,584** | 3,632 | 2 h 03 m 33 s |
| DeepFM / Criteo_x1 | `deepfm` (BARS) | 4096 | 15 | 76,538 | **95,699** | 2 h 08 m 11 s |
| DeepFM / Criteo_x1 | `deepfm` (BARS, BF16) | 4096 | 19 | 94,631 | 95,062 | 1 h 52 m 30 s |
| DCN-Mix / Criteo_x1 | `dcn_mix` (BARS) | 4096 | 18 | **108,573** | 108,385 | 1 h 33 m 27 s |
| DCN-Mix / Criteo_x1 | `dcn_mix` (BARS, BF16) | 4096 | 18 | 91,842 | 92,499 | 1 h 50 m 04 s |
| LightGCN / Gowalla, 2 cards | `lightgcn` (code) | 1024/rank | 1000 | **35,730** | 35,770 | 6 h 29 m 32 s |

The DeepFM FP32 mean is pulled down by epochs 8–13, which shared the card with
another job (42–47 k vs ~103 k samples/s); its median is the fair number, and
it equals the BF16 median.  The controlled 1-epoch probes put BF16 at
87,758 (DeepFM) and 86,505 (DCN-Mix) samples/s vs 99,295 and 105,882 for FP32 —
i.e. BF16 training is 12–18 % slower.  The DCN-Mix full-run pair is clean and
confirms it (−15 %).  The 2-card LightGCN row is the first CNCL DDP run on node
B; node A's 2-card paper-protocol row (63,700) uses half the per-rank batch and
`index_add` instead of the default `sparse` propagation.

Inference, same definition, `--device mlu:0`, batch 8192:

| Model | Dataset | Rows per pass | FP32 s/pass | FP32 rows/s | BF16 s/pass | BF16 rows/s |
| --- | --- | --- | --- | --- | --- | --- |
| DeepFM | Criteo_x1 test (4,587,167) | 4,587,167 | 4.44 | **1,032,959** | 5.87 | 781,934 (−24.3 %) |
| DCN-Mix | Criteo_x1 test (4,587,167) | 4,587,167 | 4.72 | **972,714** | 6.17 | 744,244 (−23.5 %) |
| SASRec | MovieLens-1M test (6,040 users × 101 candidates) | 6,040 | 0.0243 | 249,935 | 0.0227 | **267,840** |

`autocast_active=true` for every BF16 pass.  BF16 helps SASRec and hurts both
ads models on these kernels; on the P800 the same DeepFM bench gains 18 % in
BF16, so the vendor kernel mix moves the sign.  The matching MLU training
numbers and the exact commands are in
[`CAMBRICON.md`](CAMBRICON.md#inference-throughput).

## BF16 status

| Item | Status |
| --- | --- |
| Inference autocast | works (`--dtype bf16`) on P800 and MLU; 30–66 % slower on P800, and 24 % slower on MLU for DeepFM/DCN-Mix (SASRec is 7 % faster on MLU) |
| Training autocast | works; one full epoch of the 2.6 M rows at batch 32: **3,243 samples/s** (804 s), epoch-1 gAUC 0.8230 vs 0.8245 for FP32 — numerically indistinguishable |
| BF16 vs FP32 training, P800 | BF16 looks ~12–18 % faster at batch 32, but the FP32 reference is a 50-epoch mean under heavy contention (1,069–3,314 samples/s spread); a controlled same-load pair has not been run |
| BF16 vs FP32 training, MLU | controlled same-card pairs: **12–18 % slower** (DeepFM 87,758 vs 99,295; DCN-Mix 86,505 vs 105,882 samples/s) |

Two full MLU runs settle the accuracy question: DeepFM ends at AUC 0.813804 /
logloss 0.438037 vs 0.813723 / 0.438119 in FP32, and DCN-Mix at 0.813678 /
0.438308 vs 0.813739 / 0.438238 — every delta below 1e-4.  BF16 is
accuracy-safe for these models on this stack.  Speed is a different story:
BF16 training and inference are both slower on MLU (the DeepFM/DCN-Mix pair),
while P800 DeepFM inference gains 18 %.  The direction is set by the vendor
kernel mix, not by the model.

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
