# BF16 runs

Record of every BF16 training run, with the artifact path and command, so each
number in [EFFICIENCY.md](EFFICIENCY.md)'s BF16 table can be traced back to a
run.  The paper-protocol configs stay `fp32`; BF16 is a launch-time override.

## What "BF16" means here

`train.model.torch_dtype=bf16` wraps the forward pass and the loss in
`torch.autocast`: activations and matmuls run in bf16, **the master weights,
optimizer state and gradient accumulation stay fp32**.  It is mixed precision,
not a bf16 checkpoint.  Both trainers (`train_ads.py`, `train_rec.py`) take the
value from `model.torch_dtype`; `bench_inference.py --dtype bf16` uses the same
switch for inference.

```bash
# training
bash examples/run-adrec-kunlunxin.sh ads deepfm_paper 6 train.model.torch_dtype=bf16
# inference
python tools/bench_inference.py --task ads --checkpoint <run>/checkpoint \
  --data <dataset> --batch-size 8192 --dtype bf16
```

Outputs follow the usual layout: `<ADREC_OUT_ROOT>/<module>/ckpts/<name>-cuda/checkpoint/`
with `metrics.json`, `history.json`, `model.pt`.

## Runs

Batch, epochs and every other hyperparameter are those of the FP32 counterpart
named in the config column — the only change is the dtype.

| Run (log dir under `/data2/jiangxuhui/logs/adrec-runs/`) | Config | Epochs (best) | Test result | Mean samples/s |
| --- | --- | --- | --- | --- |
| `bf16-deepfm-paper` | `deepfm_paper` (Criteo_x1) | 6 (1) | AUC 0.804005 / logloss 0.447782 | 159,192 |
| `bf16-dcnmix-paper` | `dcn_mix_paper` (Criteo_x1) | 3 (3) | AUC 0.810482 / logloss 0.441042 | 15,912 |
| `bf16-dcnmix-ml1m` | `dcn_mix_ml1m_paper` (MovieLens-1M) | 32 (7) | AUC 0.850858 / logloss 0.368152 | 7,248 |
| `bf16-sasrec-paper` | `sasrec_paper` (MovieLens-1M) | 159 (139) | HR@10 0.821523 / NDCG@10 0.594037 | 8,807 |
| `bf16-sasrec-beauty` | `sasrec_beauty_paper` (Amazon Beauty) | 324 (304) | HR@10 0.468177 / NDCG@10 0.310388 | 11,456 |
| `bf16-sasrec-games` | `sasrec_games_paper` (Amazon Games) | 182 (162) | HR@10 0.7324 / NDCG@10 0.5266 | 12,635 |
| `bf16-sasrec-steam` | `sasrec_steam_paper` (Steam) | 78 (58) | HR@10 0.8593 / NDCG@10 0.6152 | 13,525 |
| `bf16-din-paper` | `din_paper` (Amazon Electronics) | 50 (7) | gAUC 0.862489 / logloss 0.4683 | 5,705 |
| `bf16-din-ml20m` | `din_ml20m` (MovieLens-20M) | 9 (8) | gAUC 0.984039 / logloss 0.1445 | 7,843 |
| `bf16-dssm` | `dssm` (MovieLens-1M) | 10 | Hit@100 0.199007 / NDCG@100 0.046237 | 174,542 |
| `bf16-dssm-ml20m` | `dssm_ml20m` (MovieLens-20M) | 10 | Hit@100 0.162680 / NDCG@100 0.038281 | 179,481 |
| `tt-bf16` | `two_tower` (synthetic) | 1 | loss 0.6923 / train 0.4402 | 1,006 |
| `bf16-lightgcn-gowalla` | `lightgcn_paper` (Gowalla) | running (1000) | — | 3,225 |
| `bf16-lightgcn-yelp` | `lightgcn_yelp2018_paper`, batch 2048 | running (1000) | — | 4,215 |
| `bf16-lightgcn-yelp-b1024` | `lightgcn_yelp2018_paper`, batch 1024 | running (1000) | — | 2,111 |

The FP32 counterpart of each row, the deltas and the inference numbers are in
[EFFICIENCY.md](EFFICIENCY.md); the per-model FP32 reports are in
[ads/results](ads/results/README.md) and [rec/results](rec/results/README.md).

## What the runs establish

* **Accuracy is unchanged where the trajectory is deterministic.**  DeepFM is
  identical to five decimals (0.804005 vs 0.8040), DCN-Mix on Criteo moves by
  0.0001, DIN on MovieLens-20M by 0.0001, and LightGCN is **bit-identical** at
  every evaluated epoch (Gowalla e40/60/80/100 = 0.1487 / 0.1569 / 0.1626 /
  0.1668 for both precisions; Yelp2018 the same up to e180).
* **The larger differences are early-stopping drift, not precision loss.**
  SASRec on Beauty ends at epoch 304 instead of 275 (+0.0078 HR@10), DSSM on
  MovieLens-1M at 0.1990 instead of 0.1921 (+0.0069), and DIN on Amazon peaks at
  epoch 7 instead of 11 (−0.0020, because the FP32 curve has a transient dip at
  epoch 4 that the BF16 curve does not).  Each is the size of a seed change.
* **Speed is kernel-dependent, not uniformly better.**  Batch 128 is 12 %
  slower, batch 512/4096 and the wide Criteo matmuls are faster, and LightGCN is
  exactly flat because the cost is the full-graph `index_add` propagation.
  SASRec's BF16 training gain depends on the dataset's sequence length: 1.6× on
  Beauty (max_len 50), 2.0× on Games (50) and 2.2× on Steam (50), against 1.6×
  on MovieLens-1M (max_len 200).  Its *inference* is ~20 % slower on the
  50-item datasets but 66 % slower on the 200-item one.

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD
# one example per module; every row above uses the same pattern
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=deepfm_paper train.model.torch_dtype=bf16 \
  train.hardware.device_type=cuda experiment.runner.nproc_per_node=1
python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=sasrec_paper train.model.torch_dtype=bf16 \
  train.hardware.device_type=cuda experiment.runner.nproc_per_node=1
```
