# DCN-Mix — MovieLens-1M, paper protocol

| Field | Value |
| --- | --- |
| Status | **finished** (32 epochs, best epoch 7) + a 5-arm configuration ladder |
| Started / finished | 2026-11-04 14:06 → 14:45 (CST) |
| Wall clock | 39 min (2,331 s) |
| Config | `examples/ads/conf/train/dcn_mix_ml1m_paper.yaml` |
| Dataset | `examples/ads/prepare_movielens1m_ctr.py` → `/data2/jiangxuhui/datasets/movielens_1m_ctr` |
| Code commit | `dc5cb07` + the `loss: mse` option (this run) |
| Device | Kunlunxin P800, card 6, vendor torch 2.9.0+cu128, fp32 |
| Parameters | 1,139,123 (7 tables × 30 dims + cross + DNN) |
| Output | `/data2/jiangxuhui/logs/adrec-runs/dcnmix-ml1m/ads/ckpts/dcn_mix_ml1m_paper-cuda/checkpoint/` |
| Command | `run-adrec-kunlunxin.sh ads dcn_mix_ml1m_paper 6` |

## Protocol (DCN-V2, WWW 2021, arXiv:2008.13535)

Target, Table 6 (mean of 5 runs): **DCN-Mix on MovieLens-1M, logloss 0.3160 /
AUC 0.8964** (110K parameters, 210K FLOPS).

| Setting | Value | Source |
| --- | --- | --- |
| dataset | MovieLens-1M ratings; 3-star removed, 1-2 → 0, 4-5 → 1 | `[paper]` §7.1.1 |
| rows | 739,012 (591,210 / 73,901 / 73,901), 7 categorical fields | `[paper]` Table 3 ("740k samples, 7 features, 3.5k vocab") |
| features | user_id, gender, age, occupation, zip, movie_id, first genre | `[ours]` the paper lists 7 features but not their names |
| split | random 80/10/10, seed 2020 | `[paper]` (random); seed `[ours]` |
| embedding size | 30 | `[paper]` §7.1.3 |
| batch size | 128 | `[paper]` §7.1.3 |
| structure | parallel CrossNetMix ‖ DNN | `[paper]` §7.2 ("parallel worked better on MovieLens-1M") |
| cross layers / experts / rank | 3 / 4 / 64 | `[ours]` the Criteo best setting; ML-1M's is not published |
| DNN | [512, 512], ReLU | `[ours]` the Criteo (2, 512); ML-1M's is not published |
| optimizer | Adam 5e-4, He Normal, bias 0, clip 10, EMA 0.9999 | `[paper]` §7.1.3 (rate `[ours]`, paper narrows 1e-4..5e-4) |
| embedding L2 | 1e-5 | `[ours]` paper searches λ ∈ {0, 3e-5, 1e-4} |
| steps | 32 epochs × 4,619 = 147,808 | `[paper]` search includes 150k |
| loss | BCE on logits | `[ours]` the paper says "treated as a regression problem"; MSE tested below |
| seed | 2021 | `[ours]` |

## Results

| Metric | This run (best epoch 7) | Paper Table 6 | Δ |
| --- | --- | --- | --- |
| test AUC | **0.851278** | 0.8964 | **−0.0451** |
| test logloss | 0.368003 | 0.3160 | +0.0520 |
| valid AUC | 0.849603 (peak) | — | — |

The run peaks at epoch 7 and then overfits (train loss 0.30 → 0.15 while
validation falls to 0.82), so the checkpoint is the epoch-7 one.

## Configuration ladder (why 0.045 is not a tuning mistake)

Six arms, everything identical except the named key:

| Arm | test AUC | test logloss | best epoch | params |
| --- | --- | --- | --- | --- |
| paper protocol (Adam 5e-4, L2 1e-5, BCE) | 0.8513 | 0.3680 | 7 | 1,139,123 |
| Adam 1e-4 | 0.8487 | 0.3733 | 12 | 1,139,123 |
| L2 1e-4 | **0.8540** | 0.3642 | 8 | 1,139,123 |
| MSE loss (regression, as §7.1.1 says) | 0.8498 | 0.3679 | 8 | 1,139,123 |
| MSE + L2 1e-4 | 0.8522 | 0.3646 | 9 | 1,139,123 |
| small structure (l=2, r=16, DNN 64²) + L2 1e-4 | 0.8529 | **0.3631** | 11 | 469,537 |

The ladder moves the metric by at most **+0.0027** (L2 1e-4) and the best arm is
still 0.0424 AUC / 0.047 logloss from the paper.  The learning rate, the loss
function (BCE vs the paper's stated regression) and the model size (1.14 M vs
0.47 M parameters) are therefore **not** what separates the two numbers.

## Sanity check: is the task learnable at all?

A "movie popularity" score (train count per movie_id) reaches test AUC
**0.5907**, user-popularity 0.4578, genre 0.4791, occupation 0.4966.  The model
at 0.851 is well above every single-field baseline, so it learns genuine
user × item interactions; the gap to 0.8964 is not a collapsed pipeline.

## What could explain the remaining 0.042

The paper does not publish, for MovieLens-1M: the seven feature names, how
multi-valued genres are encoded, or the per-dataset "best setting" (Table 6
gives that column for Criteo only).  Candidate explanations, none testable
without guessing:

* **feature encoding.**  Ours uses the first genre as a single category and
  user_id/movie_id as full-vocabulary categoricals (total vocab 13,179).  The
  paper reports 3.5k, which is close to the movie vocabulary alone — their
  seven fields may be built differently (e.g. genre multi-hot, or binned user
  features), which would change the input representation entirely.
* **per-dataset structure.**  Our cross/DNN shape is the Criteo best setting;
  the paper's ML-1M DCN-Mix has 110K non-embedding parameters and 210K FLOPS,
  i.e. much smaller than any arm above (the 469K-parameter arm is the closest).
* **5-run mean vs one seed.**  The paper averages five runs; a single seed
  explains a few 1e-3, not 0.04.

## Artifacts

| Path | Contents |
| --- | --- |
| `.../dcnmix-ml1m/ads/ckpts/dcn_mix_ml1m_paper-cuda/checkpoint/` | main run: model.pt (4.3 MB), history.json, metrics.json |
| `.../dcnmix-ml1m-{lr1e4,l2e4,mse,mse-l2e4,small}/...` | the five ladder arms |

## Reproduce

```bash
export FLAGOS_ADS_DATA=/data2/jiangxuhui/datasets
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python examples/ads/prepare_movielens1m_ctr.py
python -m flagscale.run --config-path examples/ads/conf --config-name train \
  action=test train=dcn_mix_ml1m_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```
