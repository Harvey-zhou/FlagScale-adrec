# SASRec and LightGCN paper protocols

How `examples/rec/conf/train/sasrec_paper.yaml` and `lightgcn_paper.yaml` are
built and why every value is what it is.  These are separate from `sasrec.yaml`
and `lightgcn.yaml`, which follow the reference *code* defaults; the paper
configs follow the paper where it speaks and the code only where it is silent.

Sources:

* SASRec — Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018,
  arXiv:1808.09781v1; code `kang205/SASRec` (TensorFlow) and the endorsed
  PyTorch port `pmixer/SASRec.pytorch`.
* LightGCN — He et al., *LightGCN: Simplifying and Powering Graph Convolution
  Network for Recommendation*, SIGIR 2020, arXiv:2002.02126v4; code
  `kuandeng/LightGCN` (TensorFlow) and `gusye1234/LightGCN-PyTorch`.
* LightGCN's early stopping is defined by reference to NGCF (Wang et al.,
  SIGIR 2019, arXiv:1905.08108), §4.2.3.

Every value is tagged with its source:

| Tag | Meaning |
| --- | --- |
| `[paper]` | stated in the paper; must not change in a paper-protocol run |
| `[code]` | not in the paper; taken from the authors' released code |
| `[ours]` | neither specifies it; our choice, free to tune |

## 1. Datasets

Both datasets are the authors' released splits, verified against the paper
statistics (see [EVIDENCE.md](EVIDENCE.md)):

| Dataset | Contents | Used by |
| --- | --- | --- |
| `movielens_1m_sasrec` | 6,040 users / 3,416 items / 999,611 interactions; 5-core; leave-one-out (train = `k=1..n-2`, validation = `n-1`, test = `n`); `max_len=200`; 1 positive + 100 fixed candidate negatives per evaluation | SASRec |
| `gowalla_lightgcn` | 29,858 users / 40,981 items; 810,128 train + 217,242 test interactions (the NGCF/LightGCN split); no validation split | LightGCN |

## 2. SASRec

| Setting | Value | Source | Config key |
| --- | --- | --- | --- |
| optimizer | Adam | `[paper]` IV-C | (trainer default) |
| learning rate | 1e-3 | `[paper]` IV-C | `optimizer.lr` |
| Adam β1 / β2 | 0.9 / 0.98 | `[code]` (`beta2` is not in the paper) | `optimizer.betas` |
| weight decay / L2 | 0 | `[paper]`/`[code]` (`l2_flag = 0`) | `optimizer.weight_decay` |
| batch size | 128 | `[paper]` IV-C | `batch_size` |
| warmup | none | `[ours]` neither source uses it | — |
| LR schedule | constant | `[paper]`/`[code]` (no schedule) | — |
| gradient clipping | none | `[ours]` neither source uses it | — |
| self-attention blocks | 2 | `[paper]` IV-C | `model.num_blocks` |
| heads | 1 | `[paper]` IV-F (two heads was worse) | `model.num_heads` |
| latent size | 50 | `[code]` (paper Fig. 2: d ≥ 40 is satisfactory) | `model.hidden_size` |
| dropout | 0.2 | `[paper]` IV-C (MovieLens-1m) | `model.dropout` |
| max sequence length | 200 | `[paper]` IV-C | dataset `meta.json` (`max_len`) |
| item embeddings | shared between input and prediction | `[paper]` IV-C | `sasrec.py` |
| positional embedding | learned | `[paper]` III-A | `sasrec.py` |
| loss | BCE on each non-padded position, one fresh negative per position per epoch | `[paper]` III-E | `losses/rec.py`, `SequentialTrainDataset` |
| early stopping | stop after 20 consecutive epochs without validation improvement | `[paper]` IV-B | `early_stop_patience: 20` |
| evaluation | HR@10 / NDCG@10, 1 positive + 100 negatives, validation every epoch | `[paper]` IV-D | `eval_every: 1`, `select_metric: ndcg@10` |
| epoch cap | 1,000 (safety only) | `[ours]` the paper has no count | `epochs` |
| seed | 42 | `[code]` | `seed` |

The paper's Table III target on ML-1M is **Hit@10 0.8245 / NDCG@10 0.5905**.
Training negatives are resampled on every access, so each epoch sees new ones,
as in §III-E ("In each epoch, we randomly generate one negative item for each
time step").

## 3. LightGCN

| Setting | Value | Source | Config key |
| --- | --- | --- | --- |
| optimizer | Adam | `[paper]` 4.1.2 | (trainer default) |
| learning rate | 1e-3 | `[paper]` 4.1.2 "default learning rate of 0.001" | `optimizer.lr` |
| batch size | 1,024 | `[paper]` 4.1.2 "default mini-batch size of 1024" | `batch_size` |
| weight decay (Adam) | 0 | `[paper]` the only regulariser is the L2 term in Eq. (15) | `optimizer.weight_decay` |
| L2 on layer-0 embeddings | 1e-4 | `[paper]` 4.5: optimal λ for Gowalla | `optimizer.bpr_weight_decay` |
| warmup / LR schedule / grad clipping | none | `[ours]` neither source uses them | — |
| embedding size | 64 | `[paper]` 4.1.2 | `model.embedding_dim` |
| layers | 3 | `[paper]` 4.1.2 / Table 3 | `model.num_layers` |
| layer combination | uniform 1/(K+1) | `[paper]` 3.1.2 | `lightgcn.py` |
| propagation | symmetric D^-1/2 A D^-1/2, no self-loops | `[paper]` Eq. (3)/(7) | `lightgcn.py` |
| initialisation | Xavier (Glorot) | `[paper]` 4.1.2 (the code uses N(0, 0.1)) | `model.init: xavier` |
| loss | BPR + λ‖E(0)‖² | `[paper]` Eq. (15) | `losses/rec.py` |
| epochs | 1,000 | `[paper]` 4.1.2 "typically 1000 epochs are sufficient" | `epochs` |
| evaluation | all-ranking recall@20 / ndcg@20, training items masked, every 20 epochs | `[paper]` 4.1/Fig. 3 | `eval_every: 20`, `topk: [20]` |
| early stopping | disabled — see below | `[ours]` conflict, documented | `early_stop_patience: 0` |
| seed | 2020 | `[code]` | `seed` |

Paper target (Table 3, 3 layers): **recall@20 0.1823 / ndcg@20 0.1555**
(released PyTorch code: 0.1824 / 0.1547).

### Early stopping conflict

The paper says "the early stopping and validation strategies are the same as
NGCF", and NGCF §4.2.3 defines that as stopping when validation recall@20 does
not increase for 50 successive epochs.  But the distributed Gowalla split
(`train.txt` / `test.txt`, the split the paper says it received from the NGCF
authors) has **no validation file**, and the released code trains all 1,000
epochs with no early stopping.  Carving 10% out of `train.txt` would change the
training data and would no longer reproduce the published numbers.

We follow the released code: 1,000 epochs, test recall recorded every 20
epochs, final epoch reported.  The trainer supports `early_stop_patience`, so
the paper rule can be switched on the day a validation split is available.

## 4. Cost on the Kunlunxin P800 (measured 2026-10-29)

| Model | Protocol | Measured | Full run |
| --- | --- | --- | --- |
| SASRec | batch 128, eval every epoch | ~1.2 s/epoch (200 epochs in 244 s without per-epoch eval) | early stop decides; ~5–10 min |
| LightGCN, batch 1024 (paper) | 792 steps/epoch | 319 ms/step, 252.7 s/epoch | **~70 h** |
| LightGCN, batch 2048 (code) | 396 steps/epoch | 319 ms/step, 126.4 s/epoch | ~35 h |

The per-step cost is dominated by the full-graph propagation (1,620,256 directed
edges × 64 dims × 3 layers; 159 ms forward + 160 ms backward) and is
independent of the batch size, so the paper's smaller batch doubles the wall
clock.  `torch.sparse.mm` is unavailable on this runtime (`CUDA error: invalid
device function`), so propagation uses the mathematically identical
`index_select` + `index_add` path (`model.propagation: index_add`).

## 5. Disclosed deviations

1. **SASRec evaluation negatives.**  The candidate list is drawn once with a
   fixed seed and stored in the dataset; it excludes *every* item of the user.
   The reference draws 100 negatives at each evaluation and excludes only the
   training items.  Both use 1 positive + 100 negatives as the paper says.
2. **LightGCN model selection.**  Test recall doubles as the monitoring metric
   (no validation split exists); the headline is the final epoch, not the best
   epoch.  This is the released code's protocol.
3. **Propagation kernel.**  `index_add` instead of `sparse.mm`; identical math,
   required by the runtime.
4. **Initialisation.**  Xavier as the paper states; the later PyTorch code uses
   N(0, 0.1).  Use `model.init: normal` to match the code instead.
5. **Test-time negative sampling (SASRec) and test-set monitoring (LightGCN)**
   mean the reported numbers are not untouched held-out scores in the strict
   sense; both are the references' own protocols.

## 6. How to run

```bash
export FLAGOS_REC_DATA=/data2/jiangxuhui/datasets/rec
export FLAGSCALE_HOME=$PWD PYTHONPATH=$PWD

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=sasrec_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1

python -m flagscale.run --config-path examples/rec/conf --config-name train \
  action=test train=lightgcn_paper train.hardware.device_type=cuda \
  experiment.runner.nproc_per_node=1
```

Outputs: `$FLAGOS_REC_CKPTS/<name>-cuda/checkpoint/{model.pt,history.json,metrics.json,config.json}`.
`history.json` and `model.pt` are refreshed at every new best monitored epoch,
so an interrupted run still leaves the best model behind.
