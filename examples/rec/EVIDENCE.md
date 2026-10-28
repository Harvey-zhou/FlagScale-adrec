# Evidence for datasets, protocols and reference numbers (rec)

Every dataset, split, hyperparameter and target number used in `examples/rec`
is taken from one of three kinds of source: (1) the original paper, (2) the
authors' official code, (3) an authoritative framework or benchmark
implementing the model.  Numbers below were read from the linked pages;
"verified" marks facts re-checked on the downloaded data or code.

## Datasets

| Dataset | Used by | Source | Verification |
| --- | --- | --- | --- |
| MovieLens-1M | SASRec, DSSM | https://grouplens.org/datasets/movielens/1m/ (`ratings.dat`, `users.dat`, `movies.dat`) | verified: after dropping users/items with < 5 actions, 6,040 users / 3,416 items / 999,611 actions, identical to kang205/SASRec `data/ml-1m.txt` |
| Gowalla (LightGCN split) | LightGCN | kuandeng/LightGCN `Data/gowalla/{train,test}.txt`; also reczoo/Gowalla_m1 | verified: md5 of train/test/user_list/item_list identical between the two sources; 29,858 users / 40,981 items / 810,128 train / 217,242 test interactions |

## SASRec

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | Kang & McAuley, ICDM 2018, arXiv 1808.09781 | 5-core filtering, time-ordered leave-one-out, test item ranked against 100 random negatives; Table III ML-1M **Hit@10 0.8245, NDCG@10 0.5905**; Table II avg actions/user 163.5 (= (999,611 - 2 x 6,040) / 6,040, i.e. counted on the training part; the full dataset gives 165.5) |
| Official code | github.com/kang205/SASRec (TF); PyTorch port github.com/pmixer/SASRec.pytorch endorsed in the kang205 README | ml-1m: maxlen 200, dropout 0.2, hidden 50, 2 blocks, 1 head, lr 1e-3, Adam beta2 0.98, batch 128, 201 epochs, evaluation every 20 epochs; pmixer README NDCG@10 0.5897-0.5918 / HR@10 0.8190-0.8225 |
| Framework | RecBole SASRec (recbole.io) | implementation only; its default protocol (CE loss, full ranking) is not comparable |

Protocol notes: the 100 evaluation negatives are drawn once with a fixed seed
(the reference redraws them at every evaluation) and exclude every item of the
user (the reference excludes the training items).

## DSSM (two-tower retrieval)

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | Huang et al., CIKM 2013 (Microsoft Research PDF) | web-search model trained on proprietary click logs (16,510 queries); cannot be reproduced on public data |
| Official code | none published | - |
| Framework / benchmark | datawhalechina/torch-rechub `examples/matching/run_ml_dssm.py`, `torch_rechub/models/matching/dssm.py` | MovieLens-1M; user tower (user_id, gender, age, occupation, zip, mean-pooled history <= 50) and item tower (movie_id, first genre), embedding 16, towers [256, 128, 64] BN + PReLU, L2-normalized outputs, point-wise BCE with 3 popularity-sampled negatives, Adam 1e-4 / weight decay 1e-6, batch 4096, 10 epochs, seed 2022; README **DSSM Hit@100 = Recall@100 = 14.74%** |
| Framework | TorchRec `examples/retrieval` two-tower | trains on synthetic data; no accuracy published |

Caveat: the torch-rechub reference is loosely specified (the script defaults to
a sample file and prints top-10 while the README reports @100; `temperature`
is passed but not applied in `DSSM.forward`).  The run follows the code and
reports Hit/Recall/NDCG at 10, 50 and 100.

## LightGCN

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | He et al., SIGIR 2020, arXiv 2002.02126 | Gowalla 29,858 / 40,981 / 1,027,370; all-ranking recall@20 / ndcg@20; Table 3 (3 layers) **0.1823 / 0.1555**; Table 4 best 0.1830 / 0.1554 |
| Official code | kuandeng/LightGCN (TF): README Gowalla 0.18236 / 0.15539; gusye1234/LightGCN-PyTorch (linked from the paper): 3 layers 0.1824 / 0.1547 | emb 64, 3 layers, lr 1e-3, decay 1e-4, bpr_batch 2048, 1000 epochs, seed 2020; test evaluated during training, no validation split |
| Framework / benchmark | BARS `matching/gnn/LightGCN/LightGCN_gowalla_m1` | recall@20 0.1820 / ndcg@20 0.1547 (4 layers) |

Implementation note: the reference propagates with `torch.sparse.mm`.  Torch-FL
has no sparse tensor backend (`aten::_sparse_coo_tensor_with_dims_and_tensors`
is unavailable for the flagos device), so the Torch-FL run uses the
mathematically identical `index_select` + `index_add` propagation
(`model.propagation=index_add`); the native CUDA run uses `sparse`.

### More LightGCN datasets (Yelp2018, Amazon-Book)

Same recipe as Gowalla (`lightgcn_yelp2018.yaml`, `lightgcn_amazon_book.yaml`), data from
gusye1234/LightGCN-PyTorch `data/` (the authors' split).  `prepare_lightgcn.py` checks the
statistics.  Verified on the distributed files: users / items / train + test interactions =
Yelp2018 31,668 / 38,048 / 1,237,259 + 324,147 = 1,561,406, Amazon-Book 52,643 / 91,599 /
2,380,730 + 603,378 = 2,984,108, which are the totals of the paper (Table 2).

| Dataset | Paper Table 3 (3 layers) | PyTorch README (layer=3) |
| --- | --- | --- |
| Yelp2018 | recall@20 0.0639 / ndcg@20 0.0525 | 0.06347 / 0.05238 |
| Amazon-Book | 0.0410 / 0.0318 | not listed |

## Two-tower (TorchRec), config `two_tower`

| Kind | Source | What it gives |
| --- | --- | --- |
| Paper | "A Dual Augmented Two-tower Model for Online Large-scale Recommendation", DLP-KDD 2021 (as cited in TorchRec's docstring) | TorchRec's example is a simplified version of it |
| Framework | meta-pytorch/torchrec `examples/retrieval` (`modules/two_tower.py`, `two_tower_train.py`, `data/dataloader.py`), commit `042a13d` | two tables of 1024**2 x 64 (`EmbeddingBagCollection`), a ReLU MLP [128, 64] per tower, inner-product logits, BCE-with-logits, learning rate 0.01, batch 32, 100 iterations; row-wise Adagrad on the tables, Adam on the MLPs |

This **replaces torch-rechub's DSSM** among the six advertising / recommendation models (the
DSSM paper's data are not public; torch-rechub's `dssm` config stays as an alternative).

TorchRec's example trains on **random data** (`RandomRecDataset` in the MovieLens-20M format:
keys `userId` / `movieId`, `ids_per_feature=1`, no dense features), so **there is no accuracy
target**.  The dataset caches 10 batches and cycles through them, and a sample draws 0 or 1 id
per feature (an empty bag pools to zeros).  `datasets/rec.py` reproduces this draw order with
a seeded generator on the host, so every device sees the same batches.

Acceptance is therefore (1) the loss trajectory against a CUDA run of the same config and
seed, and (2) the internal check that the training loss falls below ln 2 on the cycled
batches (they are memorised) while the held-out loss stays near ln 2 = 0.693.  On CPU with
seed 42, 100 iterations: running loss 0.698 -> 0.521, training-set loss 0.440, held-out loss
0.6923; 134,250,880 parameters (= 2 x 1,048,576 x 64 + 2 x 16,576).

Deviations from the example: a native port of the model and of `RowWiseAdagrad` instead of
`DistributedModelParallel`, FBGEMM and the pipelined trainer; `nn.Embedding` times the 0/1
length instead of `EmbeddingBag` (identical for one id per feature); dense gradients for the
tables.  The FAISS index and the retrieval module are not ported.

## SASRec on Beauty, Games and Steam

Configs `sasrec_beauty`, `sasrec_games`, `sasrec_steam`; paper settings for the sparser
datasets: maximum length 50, dropout 0.5 (ML-1M: 200 and 0.2).  Targets, paper Table III
(SASRec Hit@10 / NDCG@10): Beauty 0.4854 / 0.3219, Games 0.7410 / 0.5360, Steam 0.8729 / 0.6306.

Data: the preprocessed `user item` files of kang205/SASRec (`data/Beauty.txt`, `Video.txt` for
Amazon Video Games, `Steam.txt`); pmixer/SASRec.pytorch carries files with identical md5.
Verified against the paper's Table II: Games 31,013 users / 23,715 items (exact), Steam
334,730 / 13,047 (exact), Beauty 57,289 items and 394,908 actions (the paper lists 52,024
users, the file has 52,204, which looks like a typo in the paper).

Caveat: the paper says users and items with fewer than 5 actions are discarded, but the files
still hold shorter sequences (11,978 users of Beauty have fewer than 5).  As in the official
`data_partition`, a user with fewer than 3 interactions has no validation or test item and is
only used for training.  `prepare_sasrec_txt.py --train-negatives train` makes the training
sampler exclude only the training items, as the reference sampler does (the default `all`
matches `prepare_movielens_sasrec.py`).
