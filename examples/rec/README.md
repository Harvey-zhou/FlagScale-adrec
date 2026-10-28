# Recommendation model training with FlagScale

Native FlagScale training for the recommendation models of the FlagOS
localization project.  Each model follows the official implementation or the
framework whose result it is compared with; see [EVIDENCE.md](EVIDENCE.md).

| Config | Model | Data | Reference |
| --- | --- | --- | --- |
| `sasrec` | SASRec (kang205/SASRec protocol) | MovieLens-1M, 5-core, leave-one-out, 100 sampled negatives | paper HR@10 0.8245 / NDCG@10 0.5905 |
| `dssm` | DSSM two-tower (torch-rechub) | MovieLens-1M, last item per user, full-catalogue retrieval | torch-rechub Hit@100 14.74% |
| `lightgcn` | LightGCN (gusye1234/LightGCN-PyTorch) | Gowalla (official split) | paper recall@20 0.1823 / ndcg@20 0.1555 |
| `lightgcn_yelp2018`, `lightgcn_amazon_book` | LightGCN | Yelp2018, Amazon-Book (official split) | paper recall@20 0.0639 / 0.0410 |
| `sasrec_beauty`, `sasrec_games`, `sasrec_steam` | SASRec | Amazon Beauty, Amazon Games, Steam (leave-one-out, 100 sampled negatives) | paper HR@10 0.4854 / 0.7410 / 0.8729 |
| `two_tower` | TorchRec `examples/retrieval` two-tower | random data in the MovieLens-20M format | none (random data); see EVIDENCE.md |

`two_tower` replaces `dssm` in the six-model set; `dssm` remains available.

Layout (independent of the retrieval and advertising modules):

* `flagscale/models/rec/`: `sasrec.py`, `dssm.py`, `lightgcn.py`, `two_tower.py`, shared layers, registry;
* `flagscale/train/datasets/rec.py`, `losses/rec.py`, `metrics/rec.py`;
* `flagscale/train/train_rec.py`: per-task protocol (see its docstring);
* data under `$FLAGOS_REC_DATA` (default `/flagos-rec/datasets`), results under
  `$FLAGOS_REC_CKPTS` (default `/flagos-rec/ckpts`).

## Data

```bash
python examples/rec/prepare_movielens_sasrec.py   # -> /flagos-rec/datasets/movielens_1m_sasrec
python examples/rec/prepare_movielens_dssm.py     # -> /flagos-rec/datasets/movielens_1m_dssm
python examples/rec/prepare_gowalla.py            # -> /flagos-rec/datasets/gowalla_lightgcn
# LightGCN-PyTorch/data/<dataset>/{train,test}.txt
python examples/rec/prepare_lightgcn.py --dataset yelp2018 --raw <LightGCN-PyTorch>/data/yelp2018
python examples/rec/prepare_lightgcn.py --dataset amazon-book --raw <LightGCN-PyTorch>/data/amazon-book
# SASRec/data/{Beauty,Video,Steam}.txt (Video is Amazon Games)
python examples/rec/prepare_sasrec_txt.py --dataset beauty --input <SASRec>/data/Beauty.txt
```

`two_tower` needs no data: the batches are generated from the seed.

## Run

```bash
export PYTHONPATH=$PWD FLAGSCALE_HOME=$PWD WANDB_MODE=offline
# Torch-FL (LightGCN needs index_add propagation: Torch-FL has no sparse tensors)
FLAGSCALE_PYTHON_ENV=/flagos-search-codes/torch-fl-cuda/.venv \
python -m flagscale.run --config-path examples/rec/conf --config-name train action=test train=lightgcn \
  train.hardware.device_type=flagos train.hardware.distributed_backend=flagos train.model.propagation=index_add
# native CUDA baseline
FLAGSCALE_PYTHON_ENV=/opt/conda \
python -m flagscale.run --config-path examples/rec/conf --config-name train action=test train=lightgcn \
  train.hardware.device_type=cuda
```

Kunlunxin (vendor torch, which reports the card as `cuda`):

```bash
VENDOR_ENV_SH=<script that activates the vendor torch> \
  bash examples/run-adrec-kunlunxin.sh rec lightgcn_yelp2018 0   # cards: 0 or 0,1
```

Outputs: `$FLAGOS_REC_CKPTS/<model>-<device>/checkpoint/{model.pt,config.json,metrics.json}`
and an offline W&B run under `.../checkpoint/wandb`.
`train.hardware.pin_memory` (or `FLAGSCALE_PIN_MEMORY`) turns page-locked host memory off
for a runtime that does not support it.

`python examples/rec/smoke.py --model sasrec --device flagos` runs three steps
without FlagScale; `--config <name>` selects any config in `conf/train/`.
