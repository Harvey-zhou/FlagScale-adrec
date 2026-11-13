# Cambricon MLU runs (ads / rec)

The ads and rec models run on Cambricon cards through the vendor stack:
`torch_mlu` + CNCL, driven by
[`flagscale/platforms/platform_mlu.py`](../flagscale/platforms/platform_mlu.py)
(`device_type=mlu`, backend `cncl`) and launched with
[`run-adrec-cambricon.sh`](run-adrec-cambricon.sh).  Torch-FL is not involved;
its Cambricon backend is a separate track.  Model, loss, data and metric code
are the ones used on CUDA.

Status: adapter, launcher and container wrapper written.  Two MLU590-M9 nodes
are in use (see "Two MLU nodes" below).  Node A (2026-11-03 / 11-06 / 11-07 /
11-09): preflight, the `ads` smokes, single-card `rec` training (`sasrec`,
`dssm`, `two_tower`), the four paper-protocol SASRec runs (ML-1M, Beauty,
Games, Steam), 2-card CNCL DDP on ML-1M, and the full LightGCN Gowalla and
Yelp2018 runs (paper and code settings).  Node B (2026-11-09 / 11-11): the same
`ads` and `rec` smokes, full `ads` `deepfm_paper` / `dcn_mix_paper` /
`din_paper_dice` / `din_dice_author` and the three BARS `ads` runs (DeepFM and
DCN-Mix also in BF16), `rec` `sasrec_paper` / `dssm` / 2-card code-settings
`lightgcn` on Gowalla, and the 1,000-epoch `lightgcn_amazon_book` run (in
progress).  `ads din_paper` finished on node A (2026-11-11).
Multi-node is still TODO.

## Two MLU nodes

Two MLU590-M9 hosts are used, and a number has to name the node it came from.
Hostnames and addresses are omitted on purpose: this file lives in a public
repository.

| Label | Role | MLU driver | Recorded in |
| --- | --- | --- | --- |
| node A | 2026-11-03..11 `ads` smokes, full `rec` runs (SASRec / DSSM / LightGCN), 2-card DDP, BF16 batch, `ads din_paper` | 6.5.52 | Cluster environment, Host inventory, operator probe and training results |
| node B | 2026-11-09..11 `ads` + `rec` smokes, full `ads` runs (BARS + paper, BF16 pair), `rec` DSSM / SASRec / code-settings LightGCN, `lightgcn_amazon_book` in progress | 6.2.29 | Operator probe results, training results and the `*_mlu.md` run records |

Both have the same CPU, RAM and card layout (2× Hygon C86 7390, 2 TiB RAM,
8× MLU590-M9); the MLU driver is what tells them apart.

## Cluster environment (node A)

This table is node A.  Most rows come straight from the preflight log,
`<ADREC_OUT_ROOT>/<module>/logs/preflight-<config>.log` (one per config so that
concurrent launches do not overwrite each other; `preflight.log` before 2026-11-04)
(`PREFLIGHT_ONLY=1 bash examples/run-adrec-cambricon.sh ads din 0`).  The
host, driver and image rows were verified on 2026-11-03 with `cnmon`,
`modinfo` and a read of the vendor bundle.

| Item | Value | Source |
| --- | --- | --- |
| Cluster / host | single node, 8× MLU590-M9 | `cnmon` |
| Card model and memory | MLU590-M9, 98304 MiB (96 GiB) each | `cnmon` |
| Driver | 6.5.52, firmware v1.5.0 | `cnmon`, `modinfo` |
| NEUWARE / CNToolkit, CNNL, CNCL versions | host: no NEUWARE; image: CNToolkit 4.7.3 at compile time; runtime CNRT 7.7.3, CNNL 2.2.18, CNNLExtra 2.5.0, CNCL 1.30.10, MLU-OPS 1.13.0 | preflight.log (torch_mlu version banner) |
| Container image | `cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312` (loaded from the vendor tarball) | `docker images`, vendor bundle |
| Python | 3.12.3 in the image (`/torch/venv3/pytorch`), 3.8.10 on the host | preflight.log |
| torch / torch_mlu | 2.12.1+cpu / 1.34.1+torch2.12.1, exercised on card 0 | preflight.log |
| fp64 on card | yes | preflight.log |
| Page-locked host memory | yes, so it is on by default for `mlu` (`FLAGSCALE_PIN_MEMORY=false` turns it off) | preflight.log |
| CNCL available | yes (`torch.distributed.is_cncl_available()`) | preflight.log |
| `ADREC_VENV` (or env activation script) | not needed: `/torch/venv3/pytorch` inside the 26.08 container is activated by the login shell and is also a launcher default | preflight.log |
| Data root (`ADREC_DATA_ROOT`), md5 checked against EVIDENCE.md | `/srv/jiangxuhui/datasets`; `AmazonElectronics_x1` raw md5 `e9bf80b9…` / `57a20e82…` and `Criteo_x1` `30b89c1c…` / `f73c71fb…` / `2c48b26e…` match EVIDENCE.md; `SASRec` Beauty/Games/Steam `.txt` come from the authors' repo (via `api.github.com`) and their row/count statistics match `PAPER_PROTOCOLS.md`; encoded with the `prepare_*.py` scripts | md5sum, prepare output |

## Host inventory (node A)

Verified 2026-11-03.  Hostnames and network addresses are left out on purpose:
this file lives in a public repository.

| Item | Value |
| --- | --- |
| Node | 1 node, 8× MLU590-M9 |
| OS / kernel | Ubuntu 20.04.6 LTS, 5.15.0-139-generic |
| CPU | 2× Hygon C86 7390 (32 cores / 64 threads each), 128 threads, 8 NUMA nodes |
| RAM | 2003 GiB |
| MLU | 8× MLU590-M9, 96 GiB each, all healthy, ECC errors 0 |
| MLU PCIe | Gen4 x16 per card, replay counter 0 |
| MLU clock | 1600 MHz default.  Idle cards report 1200 MHz with `Clock Setting: Active` (a power-saving state); under real load the setting clears and the clock reaches 1600 MHz — sampled during a LightGCN step on 2026-11-09 (util 66%, TDP Not Active).  A dense-matmul stress instead hit the 550 W TDP cap and settled at ~1236 MHz, so power-heavy kernels are power-limited, not clock-limited |
| Driver / firmware | 6.5.52 (runfile install, DKMS-built), firmware v1.5.0, `mlumsgd` running |
| Storage | `/` 437 GB (109 GB free), `/srv` 11 TB (5.9 TB free) |
| Network | 11 mlx5 HCAs, RoCE interfaces up |
| Internet / mirrors | `huggingface.co` times out; `hf-mirror.com` works (`HF_ENDPOINT=https://hf-mirror.com`); GitHub `raw.githubusercontent.com` is blocked but `api.github.com` works (contents API with `Accept: application/vnd.github.raw` supports Range) |
| Docker | installed; the login user is in the `docker` group (takes effect on a new login; `sg docker -c …` works in a running shell), set `DOCKER=docker` to skip sudo |
| Vendor bundle | shared storage: drivers 6.2.29 and 6.5.52, `.deb` packages, cp312/pt2.12.1 wheels, five container tars, SDK docs |
| Open items | stale `cambricon-mlu-driver-dkms 6.2.29` dpkg/dkms record; no MagicMind (inference only, not needed for training); multi-node is unvalidated — CNCL reports "No available IB NIC" and there is no `/etc/cncl.conf`, so only single-node MLU-Link has been exercised |

## Entering the container

The host's Python is 3.8 and FlagScale needs >= 3.10, so every run goes through
the 26.08 image; `run-adrec-cambricon.sh` refuses to start on an older Python.
[`cambricon-container.sh`](cambricon-container.sh) loads the image when needed,
keeps one container per user (`adrec-cambricon-$USER`) and runs a command in it:

```bash
export CAMBRICON_IMAGE_TAR=<vendor bundle>/cambricon_pytorch_container-v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312.tar.gz
export WORK_ROOT=/srv/<user>           # data and outputs; `/` has ~100 GB free
PREFLIGHT_ONLY=1 bash examples/cambricon-container.sh bash examples/run-adrec-cambricon.sh ads din 0
bash examples/cambricon-container.sh   # interactive shell
bash examples/cambricon-container.sh --stop
```

It runs `docker` (`sudo docker` by default; set `DOCKER=docker` when the login
user is in the docker group), passes through every `/dev/cambricon*` character device
(`PRIVILEGED=1` as a fallback), mounts this checkout, `WORK_ROOT` and, if
present, `/flagos-ads` and `/flagos-rec` at their host paths, and forwards the
launcher's variables (`PREFLIGHT_ONLY`, `SMOKE`, `ADREC_*`, ...).  Outputs
written in the container are root-owned.

| Container item | Value |
| --- | --- |
| Image reference after `docker load` | `cambricon-base/pytorch:v26.08.0-torch2.12.1-torchmlu1.34.1-ubuntu24.04-py312` (already loaded, 22.3 GB) |
| Device passthrough enough, or `PRIVILEGED=1` needed | enough: 18 `/dev/cambricon*` nodes passed through, preflight saw the card with `MLU_VISIBLE_DEVICES=0` |
| Python with torch_mlu inside the image (`ADREC_VENV` / `CAMBRICON_ENV_SH`) | `/torch/venv3/pytorch/bin/python3` (3.12.3), torch_mlu 1.34.1 |

## Order of work

1. **Preflight**: `PREFLIGHT_ONLY=1 bash examples/run-adrec-cambricon.sh ads din 0`
   (inside the container, see above).
2. **Operator probe**, per config, single card, 3 steps + 1 evaluation:
   * `SMOKE=1 ALLOW_CPU_FALLBACK=1 bash examples/run-adrec-cambricon.sh <module> <config> 0`
     lists every operator torch_mlu would run on the CPU, in one pass;
   * `SMOKE=1 bash examples/run-adrec-cambricon.sh <module> <config> 0` must
     then pass with fallback off.
   * LightGCN: probe the default `sparse` propagation first; if it fails, add
     `--propagation index_add` (smoke) / `train.model.propagation=index_add`
     (training), which is mathematically identical.
3. **Full single-card runs**, compared with the same config's Kunlunxin P800 run
   in `ads/results/` / `rec/results/` where one exists, and with the reference
   numbers in `ads/EVIDENCE.md` / `rec/EVIDENCE.md`.  No CUDA run of these
   configs is on record, so the P800 runs are the only measured comparison.
4. **Two cards** (DDP over CNCL), e.g. `bash examples/run-adrec-cambricon.sh ads deepfm 0,1`.

## Operator probe results

The `ads` rows were verified on node A (2026-11-03); the `rec` rows on node B
(2026-11-09).  The `ads` smokes also pass on node B.

| Config | Strict smoke | Ops on CPU with fallback on | Ops with no MLU kernel | Notes |
| --- | --- | --- | --- | --- |
| ads `din` | yes (exit 0, fallback off) | none | none | CNNL warns `cnnlMasked_v4` is deprecated; 3-step eval AUC 0.5667 (not a result) |
| ads `deepfm` | yes (exit 0, fallback off) | none | none | Criteo_x1 33.0M/8.25M/4.59M; 23.4M parameters |
| ads `dcn_mix` | yes (exit 0, fallback off) | none | none | Criteo_x1; 21.9M parameters |
| rec `sasrec` | yes (exit 0, fallback off) | none | none | node A; ML-1M; 211,950 parameters |
| rec `sasrec_paper` | yes (exit 0, fallback off) | none | none | node B; MovieLens-1M; 211,950 parameters; 3-step eval hr@10 0.1457 (not a result) |
| rec `dssm` | yes (exit 0, fallback off) | none | none | MovieLens-1M; 329,222 parameters; 3-step eval recall@10 0.0041 (not a result); the two-tower model of the six (torch-rechub); `two_tower` is TorchRec's synthetic-data example and optional |
| rec `two_tower` | yes (exit 0, fallback off) | none | none | node A; synthetic batches, 134.3M parameters |
| rec `lightgcn` | yes (exit 0, fallback off) | none | none | node A/B; Gowalla; 4,533,696 parameters; the default `sparse` COO propagation runs on MLU, `index_add` not needed; 3-step eval recall@20 0.0007 (not a result) |
| rec `lightgcn_amazon_book` | yes (exit 0, fallback off) | none | none | node B; Amazon-Book; 9,231,488 parameters; `sparse` propagation runs on MLU; 3-step eval recall@20 0.0003 (not a result) |

The 2026-11-09 `rec` smokes ran on node B (a second 8×MLU590-M9 node) with
driver 6.2.29, torch 2.12.1+cpu, torch_mlu 1.34.1+torch2.12.1, CNRT 7.7.3,
CNNL 2.2.18, CNCL 1.30.10 and MLU-OPS 1.13.0 (preflight logs under
`<ADREC_OUT_ROOT>/<module>/logs/`).  Data: `AmazonElectronics_x1`,
`Criteo_x1`, `MovieLens-1M` and `Gowalla`, md5-checked against the values in
`ads/EVIDENCE.md` before encoding; `Amazon-Book` (`reczoo/AmazonBooks_m1`),
md5-checked against that dataset's README and encoded to the published
LightGCN split (52,643 users / 91,599 items / 2,380,730 train / 603,378 test).

## Training results

| Config | Cards | Test metric | P800 same config | Reference | Train samples/s | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| rec `sasrec` (ML-1M, code defaults, node A) | 1 | HR@10 0.8180 / NDCG@10 0.5892 | — (P800 ran `sasrec_paper`: 0.8204 / 0.5910) | 0.8190–0.8225 / 0.5897–0.5918 (pmixer code) | 8,592 | 200 epochs, 141 s total |
| rec `sasrec_paper` (ML-1M, nodes A and B) | 1 | HR@10 0.8174 / NDCG@10 0.5895 | 0.8204 / 0.5910 (197 s) | 0.8245 / 0.5905 (paper) | 8,337–8,710 | early stop @163 (best 143); 127 s; node B file `rec/results/sasrec_paper_movielens_1m_mlu.md` |
| rec `sasrec` (ML-1M, 2-card DDP, node A) | 2 | HR@10 0.8190 / NDCG@10 0.5925 | — | 0.8190–0.8225 / 0.5897–0.5918 (pmixer code) | 9,805 | CNCL over MLU-Link; 1.14× the 1-card throughput; 123 s |
| rec `sasrec_beauty_paper` (node A) | 1 | HR@10 0.4684 / NDCG@10 0.3092 | 0.4604 / 0.3034 (2,079 s) | 0.4854 / 0.3219 (paper) | 10,838 | early stop @381; 2,058 s |
| rec `sasrec_games_paper` (node A) | 1 | HR@10 0.7266 / NDCG@10 0.5222 | 0.7340 / 0.5281 (914 s) | 0.7410 / 0.5360 (paper) | 9,305 | early stop @133; 409 s |
| rec `sasrec_steam_paper` (node A) | 1 | HR@10 0.8614 / NDCG@10 0.6167 | 0.8589 / 0.6150 (4,702 s) | 0.8729 / 0.6306 (paper) | 10,672 | early stop @108; 3,416 s |
| rec `lightgcn_paper` (Gowalla, 2-card DDP, node A) | 2 | recall@20 0.1820 / ndcg@20 0.1550 | 0.1820 / 0.1553 (70 h, 1 card) | 0.1823 / 0.1555 (paper) | 63,700 | 1,000 epochs, 3 h 43 m; batch 512/rank (global 1024); `index_add`; the 1200 MHz seen on idle cards is an idle state — the cards ran at 1600 MHz during the run (sampled 2026-11-09), so this is a full-clock number |
| rec `lightgcn` (Gowalla, code settings, 2-card DDP, node B) | 2 | recall@20 0.1818 / ndcg@20 0.1553 | — (no P800 code run; the P800 paper run is 0.1820 / 0.1553) | 0.1824 / 0.1547 (released code); paper 0.1823 / 0.1555 | 35,730 | 1,000 epochs, 6 h 29 m; batch 1024/rank (global 2048); `sparse` propagation; first CNCL DDP on node B; `rec/results/lightgcn_gowalla_mlu.md` |
| rec `lightgcn_yelp2018_paper` (paper settings, node A) | 2 | recall@20 0.0569 / ndcg@20 0.0462 | — (the P800 Yelp run did not finish) | 0.0639 / 0.0525 (paper) | 45,645 | 1,000 epochs, 7 h 58 m; the gap is the paper-vs-code settings difference, see the next row |
| rec `lightgcn_yelp2018_paper` (code settings, node A) | 2 | recall@20 0.0632 / ndcg@20 0.0522 | — | 0.06347 / 0.05238 (authors' code) | 89,619 | 1,000 epochs, 4 h 49 m; reproduces the reference to 0.0003; the Yelp data files are md5-identical to the authors' |
| ads `deepfm_paper` (node B, card 4) | 1 | AUC 0.8039 / logloss 0.4477 | AUC 0.8040 / logloss 0.4477 | BARS 0.8137 / 0.4381 | 56k–219k (shared card) | best epoch 1, early stop 6; `ads/results/deepfm_paper_criteo_x1_mlu.md` |
| ads `dcn_mix_paper` (node B, card 6) | 1 | AUC 0.8104 / logloss 0.4411 | AUC 0.8104 / logloss 0.4411 | BARS 0.8142 / 0.4378 | 14,232 | 3 epochs, best epoch 3; `ads/results/dcn_mix_paper_criteo_x1_mlu.md` |
| ads `din_paper_dice` (node B, card 7) | 1 | gAUC 0.8565 / AUC 0.8573 | gAUC 0.8575 / AUC 0.8587 | paper gAUC 0.8871 | 3,619 | 50 epochs, best epoch 5; no NaN (P800 diverged at epoch 38); `ads/results/din_paper_dice_amazonelectronics_x1_mlu.md` |
| ads `din_dice_author` (node B, card 5) | 1 | gAUC 0.8689 / AUC 0.8714 | gAUC 0.8658 / AUC 0.8681 | paper gAUC 0.8871 (authors' 0.8711) | 3,584 | 10 epochs, best epoch 3; no NaN (P800 diverged at epoch 3); `ads/results/din_dice_author_amazonelectronics_x1_mlu.md` |
| ads `din_paper` (node A, card 4) | 1 | gAUC 0.8553 / AUC 0.8570 / logloss 0.4804 | gAUC 0.8555 (P800) | paper gAUC 0.8818 | ~5,100 | 50 epochs, 7 h 19 m; best epoch 5 |
| ads `din` (BARS, node B, card 1) | 1 | gAUC 0.8821 / AUC 0.8847 | — (no P800 BARS run) | BARS gAUC 0.8835 / AUC 0.8860 | 60,828 | early stop @10, best epoch 8; `ads/results/din_amazonelectronics_x1_mlu.md` |
| ads `deepfm` (BARS, node B, card 3) | 1 | AUC 0.8137 / logloss 0.4381 | — (no P800 BARS run) | BARS 0.8137 / 0.4381 | 95,699 | early stop @15, best epoch 13; card shared for epochs 8–13; `ads/results/deepfm_criteo_x1_mlu.md` |
| ads `deepfm` (BARS, **BF16**, node B, card 3) | 1 | AUC 0.8138 / logloss 0.4380 | — | BARS 0.8137 / 0.4381 (FP32 same config 0.813723 / 0.438119) | 94,631 | 19 epochs, best 17; accuracy delta <1e-4; training ~12 % slower, inference −24 %; `ads/results/deepfm_criteo_x1_mlu.md#bf16` |
| ads `dcn_mix` (BARS, node B, card 4) | 1 | AUC 0.8137 / logloss 0.4382 | — (no P800 BARS run) | BARS full-rank 0.8142 / 0.4378 | 108,385 | early stop @18, best epoch 16; `ads/results/dcn_mix_criteo_x1_mlu.md` |
| ads `dcn_mix` (BARS, **BF16**, node B, card 4) | 1 | AUC 0.8137 / logloss 0.4383 | — | BARS full-rank 0.8142 / 0.4378 (FP32 same config 0.813739 / 0.438238) | 91,842 | 18 epochs, best 16; accuracy delta <1e-4; training −15 %, inference −24 %; `ads/results/dcn_mix_criteo_x1_mlu.md#bf16` |
| rec `dssm` (ML-1M, node A) | 1 | Recall@100 0.2123 | — | 0.1474 (torch-rechub README) | 312,074 | exact full-catalogue ranking; the reference uses Annoy, so not strictly like-for-like |
| rec `dssm` (ML-1M, node B, card 1) | 1 | hit@100 0.1599 / ndcg@100 0.0334 | — (no P800 run) | torch-rechub Hit@100 14.74% | 303,033 | 10 epochs; `rec/results/dssm_movielens_1m_mlu.md` |
| rec `lightgcn_amazon_book` (node B, card 0) | 1 | running (epoch 747/1000) | — | paper recall@20 0.0410 / ndcg@20 0.0318 | ~8,300 | best recall@20 0.0415 at epoch 740, already above the paper level; ~4.8 min/epoch |
| rec `two_tower` (synthetic, node A) | 1 | neg_loss −0.6923 | — | none published | 7,045 | 100 steps; pipeline check only |

Yelp2018 reproduction note: the two Yelp rows differ only in the reference's
own settings.  The paper text specifies Xavier init, batch 1024 and λ=1e-3
(0.0569); the released code defaults to N(0, 0.1) init, batch 2048 and
λ=1e-4, and its README reports 0.06347 — with the code settings plus the
precomputed self-loop adjacency we get 0.0632.  Checked along the way: the
Yelp `train.txt` / `test.txt` are byte-identical to the authors' files
(md5 `1b8b5d22…` / `0d57d739…`), self-loops alone add ~0.0004, init and
batch alone do not move the number, and the λ change is what closes the gap.

"P800 same config" is the Kunlunxin P800 run recorded in `ads/results/` or
`rec/results/` (fp32, vendor torch 2.9.0).  It is a like-for-like comparison
only when the training code has not changed in between:

* comparable: `deepfm_paper`, `dcn_mix_paper`, `din_paper_dice`,
  `sasrec_{beauty,games,steam}_paper`, `lightgcn_paper` (the config itself
  pins `index_add`, as on the P800).  Options added after those runs default
  to the old behaviour.
* not comparable: `din_paper`.  The P800 run predates `e463e1e` and used
  PyTorch's default PReLU slope 0.25; the current code initialises it to 0.
  Compare it with the paper's 0.8818 only.
* not checked: `sasrec_paper` (ML-1M) and `lightgcn_paper` were run at
  `da5195a`, a commit missing from this branch's history.
* no P800 run: the BARS configs (`din`, `deepfm`, `dcn_mix`), `sasrec`,
  `dssm`, `lightgcn`; compare with the reference numbers.  The P800 run of
  `lightgcn_yelp2018_paper` was still in progress on 2026-11-03.

Throughput: the clock is not fixed.  Idle cards report 1200 MHz with
`Clock Setting: Active`, but the setting clears under load and the observed
rec-training clock is 1600 MHz (dense-matmul stress can instead hit the
550 W TDP cap at ~1236 MHz).  Record the clock with every samples/s figure.

## Inference throughput

`tools/bench_inference.py --device mlu:<card>` times a pass on the MLU: the device is
resolved through the platform adapter, the timing synchronises `torch.mlu`, and
`--dtype bf16` enters `torch.autocast("mlu")` or stops the run (the result's
`autocast_active` says which).  The tool sets `ENABLE_FALLBACK_TO_CPU=0` when it is
unset, so an operator without an MLU kernel fails the run instead of being timed on
the CPU.  Definition and the other tasks: [EFFICIENCY.md](EFFICIENCY.md).

```bash
bash examples/cambricon-container.sh env PYTHONPATH=$PWD python tools/bench_inference.py \
  --device mlu:0 --task ads --checkpoint <run>/checkpoint --data <dataset> --batch-size 8192 [--dtype bf16]
```

### Probe results (2026-11-11, node A, card 0)

SASRec ML-1M checkpoint, `--task sequential`, **batch 1024** (the batch the
reference table uses), one warm-up plus three timed passes:

| dtype | autocast_active | s/pass | rows/s (mean) | rows/s (median) |
| --- | --- | --- | --- | --- |
| fp32 | false | 0.0287 / 0.0297 / 0.0301 | 204,784 | 203,379 |
| bf16 | **true** | 0.0276 / 0.0268 / 0.0268 | 223,397 (+9.1 %) | 225,399 (+10.8 %) |

The first probe used batch 8192 (249,935 / 267,840 rows/s); it is not comparable
with the P800 table's batch-1024 SASRec row, so the numbers above are the aligned
ones.

Training probe, `sasrec_paper` with `train.model.torch_dtype=bf16` (one card,
`ENABLE_FALLBACK_TO_CPU=0`, no CPU fallback, no error):

| dtype | epochs (best) | mean s/epoch | samples/s | train time | HR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 163 (143) | 0.725 | 8,337 | 126.9 s | 0.81738 | 0.58953 |
| bf16 | 146 (126) | 0.782 | 7,729 | 121.7 s | **0.81589** | **0.58953** |

`autocast_active` is the tool's own check (`torch.is_autocast_enabled("mlu")`).
For training the context was verified independently: a matmul inside
`autocast_context(device, "bf16", platform_mlu)` returns `torch.bfloat16`.

Reading: this single probe put BF16 inference ~9 % faster (mean; ~11 % median),
but it did not reproduce.  Three later runs on the same checkpoint gave medians
of 117,134 / 217,161 / 225,274 rows/s while FP32 stayed at 192k–211k (batch
table below), so ML-1M is not a BF16 speedup.  BF16 training is numerically
indistinguishable (NDCG identical, HR@10 −0.18 %) but ~8 % slower per epoch — a
212 k-parameter model is too small for the conversion overhead to pay off, and
it finishes earlier only because it early-stops at epoch 146 instead of 163.
Batch BF16 only after measuring on the target stack: it is neutral on LightGCN
and loses on SASRec.

Probe checkpoint: `<ADREC_OUT_ROOT>/rec/ckpts/sasrec_paper-mlu` under a separate
`ADREC_OUT_ROOT` (removed after recording); the figures above are from its
`metrics.json` / `history.json` and the two bench JSON lines.  The ads BF16
checkpoints stay in `/home/zhouhao/flagos-adrec-cambricon/out-bf16/`.

## BF16 batch results (2026-11-11/12)

Every BF16 training run used a separate `ADREC_OUT_ROOT`
(`flagos-adrec-cambricon-bf16`) so the FP32 checkpoints were not overwritten.
Inference reads the FP32 checkpoint of the same config, so the two columns
isolate the inference dtype.  The SASRec / DSSM / LightGCN runs are node A; the
DeepFM / DCN-Mix pair was run on node B and is marked below.

### Training (FP32 vs BF16)

| Run | FP32 metric | BF16 metric | epochs FP32 → BF16 | mean s/epoch FP32 → BF16 |
| --- | --- | --- | --- | --- |
| SASRec ML-1M | HR@10 0.81738 / NDCG 0.58953 | 0.81589 / 0.58953 | 163 → 146 | 0.725 → 0.790 |
| SASRec Games | 0.72661 / 0.52219 | 0.73376 / 0.52927 | 133 → 215 | 2.918 → 3.334 |
| SASRec Beauty | 0.46843 / 0.30915 | 0.46014 / 0.29959 | 381 → 195 | 5.108 → 5.184 |
| SASRec Steam | 0.86143 / 0.61669 | 0.86164 / 0.61690 | 108 → 108 | 29.911 → 33.263 |
| DSSM ML-1M | recall@100 0.21225 / ndcg 0.04618 | 0.19288 / 0.04477 | 10 → 10 | 13.17 → 14.01 |
| LightGCN Gowalla | recall@20 0.18197 / ndcg 0.15503 | 0.18197 / 0.15503 | 1000 → 1000 | 12.718 → 12.737 |
| LightGCN Yelp2018 (code) | 0.06324 / 0.05217 | 0.06381 / 0.05265 | 1000 → 1000 | 15.81 → 27.52 * |

\*The Yelp BF16 run shared the node with the Gowalla BF16 run and the SASRec
batch, so its epoch time is not load-matched; the controlled single-card probe
below is the fair comparison.

Reading: the SASRec metric differences track the early-stop epoch, not the
precision (Steam, the only pair that stops at the same epoch, differs by
+0.0002).  BF16 is 8–15 % slower per epoch on the small SASRec models.
LightGCN is unaffected: identical metrics on Gowalla and identical
back-to-back single-card throughput.

DSSM is the one pair with the same epoch budget and a visible metric drop
(recall@100 −0.0194, −9.1 %).  It is a single run, so it could be seed variance
rather than precision; repeat before quoting it.  Its throughput is also flat
(13.17 → 14.01 s/epoch).

### LightGCN single-card throughput (30 epochs, config batch, card 4)

| Run | batch | FP32 samples/s | BF16 samples/s |
| --- | --- | --- | --- |
| LightGCN Gowalla | 1024 | 73,215 | 73,206 |
| LightGCN Yelp2018 | 2048 | 102,081 | 102,103 |

FP32 and BF16 are the same within 0.1 %: embedding lookups and `index_add` are
not autocast kernels, and the parameters stay FP32, so BF16 neither speeds up
nor shrinks LightGCN training.

### Node B: DeepFM and DCN-Mix (2026-11-11, cards 3 / 4)

The node A batch above left the two large ads models unmeasured; node B ran them
end-to-end on the same commit, BF16 in a separate `ADREC_OUT_ROOT` (`out-bf16`):

| Model | dtype | Epochs (best) | Mean samples/s | Median | Test AUC | Test logloss | Train time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepFM | fp32 | 15 (13) | 76,538* | 95,699 | 0.813723 | 0.438119 | 7,690.5 s |
| DeepFM | bf16 | 19 (17) | 94,631 | 95,062 | 0.813804 | 0.438037 | 6,750.4 s |
| DCN-Mix | fp32 | 18 (16) | 108,573 | 108,385 | 0.813739 | 0.438238 | 5,606.5 s |
| DCN-Mix | bf16 | 18 (16) | 91,842 | 92,499 | 0.813678 | 0.438308 | 6,604.4 s |

\* The DeepFM FP32 mean includes epochs 8–13 on a shared card (42–47 k vs
~103 k samples/s); its median is the fair number.  The DCN-Mix pair is clean.
Controlled 1-epoch probes on an idle card agree with the clean pair: DeepFM
87,758 (BF16) vs 99,295 (FP32), DCN-Mix 86,505 vs 105,882 samples/s.

Accuracy: every delta is below 1e-4 AUC / 1e-4 logloss, so BF16 is safe for
these models.  Speed: BF16 training is 12–18 % slower; BF16 inference on the
same checkpoints (`--task ads`, batch 8192, card 1) is also slower —
DeepFM 1,032,959 → 781,934 rows/s (−24.3 %), DCN-Mix 972,714 → 744,244
(−23.5 %), `autocast_active=true`.  On the P800 DeepFM *gains* 18 % in BF16
inference, so the sign is set by the vendor kernel mix, not the model.

### Inference (medians; batch 1024 for SASRec / LightGCN, 8192 for DSSM and DIN)

| Model | Dataset | FP32 rows/s | BF16 rows/s | change |
| --- | --- | --- | --- | --- |
| SASRec | ML-1M | 202,731 | 117,134–225,274 | unstable |
| SASRec | Games | 301,835 | 264,298 | −12.4 % |
| SASRec | Beauty | 307,047 | 271,996 | −11.4 % |
| SASRec | Steam | 305,305 | 275,929 | −9.6 % |
| DSSM | ML-1M (full catalogue) | 975,888 | 924,686 | −5.2 % |
| DIN | AmazonElectronics (384,806 rows) | 670,636 | 660,652 | −1.5 % |
| DeepFM | Criteo_x1 (4,587,167 rows, node B) | 1,031,653 | 783,970 | −24.0 % |
| DCN-Mix | Criteo_x1 (4,587,167 rows, node B) | 974,769 | 748,296 | −23.2 % |
| LightGCN | Gowalla | 2,457,798 | 2,260,501 | −8.0 % |
| LightGCN | Yelp2018 | 2,086,298 | 2,118,588 | +1.5 % |

DSSM online-only throughput (catalogue encoding excluded): FP32 1,432,879 /
BF16 1,368,100 rows/s (medians).

BF16 inference is slower on seven of the ten rows (5–24 %).  ML-1M is the one
place where BF16 looked faster, and it does not reproduce: four runs gave
medians 117,134 / 217,161 / 225,274 rows/s while FP32 stayed at 192k–211k, i.e.
the BF16 result swings by 2× and its sign is not a result.  Yelp2018 (+1.5 %)
is within run-to-run noise on a shared node.  DIN is neutral (−1.5 %) and the
two large ads models lose ~24 %.

### What to use BF16 for

Not a throughput win on this stack: SASRec training is 8–15 % slower and its
inference is mixed; LightGCN training is identical and inference slower; DeepFM
and DCN-Mix are 12–18 % slower in training and ~24 % slower in inference.  DIN
inference is neutral (−1.5 %).  Accuracy is not the constraint — the ads full
runs stay within 1e-4 of FP32.  Keep FP32 for throughput; BF16 is only worth it
where memory or an external requirement makes it necessary.

## How the launcher keeps a run honest

* **No silent CPU fallback.** torch_mlu runs an operator it has no MLU kernel
  for on the CPU, warning once per operator, unless `ENABLE_FALLBACK_TO_CPU=0`
  (on when unset; read in torch_mlu `r1.22_pt2.1.0`,
  `csrc/aten/MLUFallback.cpp`).  The launcher sets it to 0 and also scans the
  log for the "will fall back to run on the CPU" warning, failing the run on a
  hit, so a renamed variable in a newer torch_mlu cannot let a fallback
  through.  `ALLOW_CPU_FALLBACK=1` is for the probe only.
* **No `gpu_migration`.** `torch_mlu.utils.gpu_migration` rewrites
  `torch.cuda` onto the MLU, so a run would report `cuda`.  The adapter uses the
  native `mlu` device instead.
* **One PrivateUse1 owner per process.** torch_mlu and Torch-FL both claim
  PyTorch's PrivateUse1 slot.  The adapter does not import torch_mlu when
  another backend already owns it, so a Torch-FL run on a host that also has
  torch_mlu installed is unaffected.
* **Runs only on the cards named.**  The preflight requires torch to see
  exactly as many cards as were requested, so an ignored
  `MLU_VISIBLE_DEVICES` stops the launch instead of running on cards another
  job may hold.
* **Page-locked memory on by default** (`platform_mlu.supports_pin_memory`),
  since the first preflight showed it works.  Every preflight re-checks it
  through a `DataLoader`, the same path the training code uses, and stops the
  launch if it fails; `FLAGSCALE_PIN_MEMORY=false` turns it off.

Verified on the cluster (2026-11-03): `torch.distributed.is_cncl_available()`,
page-locked host memory, fp64.  Not yet verified: `MLU_VISIBLE_DEVICES` as the
card selector (the stricter preflight above checks it on the next launch),
`torch.mlu.mem_get_info` (the `mlu0:` line of the preflight log), and the fallback
variable's name in torch_mlu 1.34.1 (no config so far has an operator without
an MLU kernel, so `ENABLE_FALLBACK_TO_CPU=0` has not been exercised; the log
scan covers it either way).
