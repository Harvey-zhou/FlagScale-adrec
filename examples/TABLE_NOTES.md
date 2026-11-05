# Notes from the shared results table

The Feishu spreadsheet "国产芯片性能效率表-发改委项目" had a 备注 (notes) column on its
性能 (quality) and 效率 (efficiency) tabs.  The column was removed on 2026-11-05
(spreadsheet revision 1374) and its text is kept here verbatim.  Every note is about
the Kunlunxin P800 column; the linked result files hold the full settings and
numbers.  "Sheet rows" are the merged rows the note covered on that date.

## 性能 (quality)

| Sheet rows | Model | Dataset | Note (verbatim) | Result file |
| --- | --- | --- | --- | --- |
| 3–7 | DIN | Amazon Electronics | 昆仑芯：按论文文字实现，batch 32，单卡；测试集选最佳轮（第7轮），之后过拟合；其他厂商为各自配置的结果，口径不同 | [din_paper_amazonelectronics_x1.md](ads/results/din_paper_amazonelectronics_x1.md) |
| 15–17 | DeepFM | Criteo Kaggle | 昆仑芯：Criteo_x1 的 7:2:1 划分（论文划分未公开）；早停于第6轮，最佳为第1轮；同划分公开基准 0.8137 / 0.4381 | [deepfm_paper_criteo_x1.md](ads/results/deepfm_paper_criteo_x1.md) |
| 19–21 | DCN-Mix | Criteo Kaggle | 昆仑芯：论文最佳设置，Criteo_x1 的 7:2:1 划分（论文划分未公开）；仅训练3轮，验证 AUC 仍在上升；同划分公开基准 0.8142 / 0.4378 | [dcn_mix_paper_criteo_x1.md](ads/results/dcn_mix_paper_criteo_x1.md) |
| 37–39 | LightGCN | Gowalla | 昆仑芯：论文协议，1000 轮，取最后一轮；第900轮评测点 0.1826 / 0.1555 | [lightgcn_paper_gowalla.md](rec/results/lightgcn_paper_gowalla.md) |
| 49–51 | SASRec | Amazon Beauty | 昆仑芯：早停于第275轮（最佳第255轮），dropout 0.5；低于论文约2.5点，原因未分离；评测候选负样本固定种子一次抽取，与参考实现不同 | [sasrec_paper_amazon_beauty.md](rec/results/sasrec_paper_amazon_beauty.md) |
| 53–55 | SASRec | Amazon Video Games | 昆仑芯：早停于第182轮（最佳第162轮），dropout 0.5；评测负样本口径同 ML-1M | [sasrec_paper_amazon_games.md](rec/results/sasrec_paper_amazon_games.md) |
| 57–59 | SASRec | Steam | 昆仑芯：早停于第78轮（最佳第58轮），dropout 0.5；评测负样本口径同 ML-1M | [sasrec_paper_steam.md](rec/results/sasrec_paper_steam.md) |
| 61–63 | SASRec | MovieLens-1M | 昆仑芯：早停于第159轮（最佳第139轮）；评测候选负样本固定种子一次抽取，与参考实现不同 | [sasrec_paper_movielens_1m.md](rec/results/sasrec_paper_movielens_1m.md) |

## 效率 (efficiency)

| Sheet rows | Model | Dataset | Note (verbatim) | Result file |
| --- | --- | --- | --- | --- |
| 3–5 | DIN | Amazon Electronics | 昆仑芯：batch 1024，训练吞吐率取9轮均值（含首轮） | [EFFICIENCY.md](EFFICIENCY.md) |
| 11–13 | DeepFM | Criteo Kaggle | 昆仑芯：训练吞吐率为6轮均值（早停）；推理吞吐率 batch 8192，BF16 推理吞吐率更快 | [EFFICIENCY.md](EFFICIENCY.md) |
| 15–17 | DCN-Mix | Criteo Kaggle | 昆仑芯：训练吞吐率由3轮逐轮训练吞吐率用时换算（约 9.4–9.8 千样本/秒），batch 512，3轮未收敛；推理吞吐率暂未测 | [EFFICIENCY.md](EFFICIENCY.md) |
| 39–41 | LightGCN | Gowalla | 昆仑芯：训练吞吐率为1000轮均值；推理吞吐率（全排序）暂未测 | [EFFICIENCY.md](EFFICIENCY.md) |
| 63–65 | SASRec | MovieLens-1M | 昆仑芯：训练吞吐率为159轮均值，batch 128；推理吞吐率 batch 1024；训练吞吐率样本为用户序列 | [EFFICIENCY.md](EFFICIENCY.md) |

## Recorded in the repo since these notes were written

* **DIN / Amazon Electronics, quality.**  The note describes the run with PReLU
  initial slope 0.25 (gAUC 0.8555, best epoch 7).  The rerun with slope 0.0
  (`e463e1e`) reaches gAUC 0.8645 at best epoch 11 and is now the `din_paper`
  record ([din_paper_amazonelectronics_x1.md](ads/results/din_paper_amazonelectronics_x1.md)).
* **DeepFM and DCN-Mix / Criteo_x1, quality.**  Besides the published BARS numbers
  the notes quote, the BARS configs were re-run on this host: DeepFM 0.8112 /
  0.4406, DCN-Mix 0.8139 / 0.4380
  ([bars_baseline_criteo_x1.md](ads/results/bars_baseline_criteo_x1.md)).  The
  DeepFM gap is regularisation: embedding L2 + BatchNorm + dropout 0.1 gives
  0.8111 ([deepfm_paper_regularisation_criteo_x1.md](ads/results/deepfm_paper_regularisation_criteo_x1.md)).
  For DCN-Mix, lr 1e-4 gives 0.8115
  ([dcn_mix_paper_lr_ablation_criteo_x1.md](ads/results/dcn_mix_paper_lr_ablation_criteo_x1.md)).
* **DCN-Mix / Criteo_x1, training throughput.**  The note's 9.4–9.8 k samples/s
  was measured while another job shared the card; the lr-1e-4 run, with no other
  tenant, gives 19,488 samples/s ([EFFICIENCY.md](EFFICIENCY.md)).
* **LightGCN, inference throughput.**  Still not measured on Gowalla; Yelp2018
  full ranking is measured (128,344 users/s, [EFFICIENCY.md](EFFICIENCY.md)).
