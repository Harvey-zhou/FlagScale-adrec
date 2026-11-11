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
`din_paper_dice`, `rec` `sasrec_paper` / `dssm`, and the 1,000-epoch
`lightgcn_amazon_book` run (in progress).  `ads din_paper` is running on node A.
Multi-node is still TODO.

## Two MLU nodes

Two MLU590-M9 hosts are used, and a number has to name the node it came from.
Hostnames and addresses are omitted on purpose: this file lives in a public
repository.

| Label | Role | MLU driver | Recorded in |
| --- | --- | --- | --- |
| node A | 2026-11-03..11 `ads` smokes, full `rec` runs (SASRec / DSSM / LightGCN), 2-card DDP, `ads din_paper` in progress | 6.5.52 | Cluster environment, Host inventory, operator probe and training results |
| node B | 2026-11-09..11 `ads` + `rec` smokes, full `ads` runs, `rec` DSSM/SASRec, `lightgcn_amazon_book` in progress | 6.2.29 | Operator probe results, training results and the `*_mlu.md` run records |

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
| rec `lightgcn_yelp2018_paper` (paper settings, node A) | 2 | recall@20 0.0569 / ndcg@20 0.0462 | — (the P800 Yelp run did not finish) | 0.0639 / 0.0525 (paper) | 45,645 | 1,000 epochs, 7 h 58 m; the gap is the paper-vs-code settings difference, see the next row |
| rec `lightgcn_yelp2018_paper` (code settings, node A) | 2 | recall@20 0.0632 / ndcg@20 0.0522 | — | 0.06347 / 0.05238 (authors' code) | 89,619 | 1,000 epochs, 4 h 49 m; reproduces the reference to 0.0003; the Yelp data files are md5-identical to the authors' |
| ads `deepfm_paper` (node B, card 4) | 1 | AUC 0.8039 / logloss 0.4477 | AUC 0.8040 / logloss 0.4477 | BARS 0.8137 / 0.4381 | 56k–219k (shared card) | best epoch 1, early stop 6; `ads/results/deepfm_paper_criteo_x1_mlu.md` |
| ads `dcn_mix_paper` (node B, card 6) | 1 | AUC 0.8104 / logloss 0.4411 | AUC 0.8104 / logloss 0.4411 | BARS 0.8142 / 0.4378 | 14,232 | 3 epochs, best epoch 3; `ads/results/dcn_mix_paper_criteo_x1_mlu.md` |
| ads `din_paper_dice` (node B, card 7) | 1 | gAUC 0.8565 / AUC 0.8573 | gAUC 0.8575 / AUC 0.8587 | paper gAUC 0.8871 | 3,619 | 50 epochs, best epoch 5; no NaN (P800 diverged at epoch 38); `ads/results/din_paper_dice_amazonelectronics_x1_mlu.md` |
| ads `din_paper` (node A, card 4) | 1 | running (epoch 1/50) | gAUC 0.8555 (P800) | paper gAUC 0.8818 | — | started 2026-11-11 |
| rec `dssm` (ML-1M, node A) | 1 | Recall@100 0.2123 | — | 0.1474 (torch-rechub README) | 312,074 | exact full-catalogue ranking; the reference uses Annoy, so not strictly like-for-like |
| rec `dssm` (ML-1M, node B, card 1) | 1 | hit@100 0.1599 / ndcg@100 0.0334 | — (no P800 run) | torch-rechub Hit@100 14.74% | 303,033 | 10 epochs; `rec/results/dssm_movielens_1m_mlu.md` |
| rec `lightgcn_amazon_book` (node B, card 0) | 1 | running (epoch 430/1000) | — | paper recall@20 0.0410 / ndcg@20 0.0318 | ~8,300 | best recall@20 0.0412 at epochs 400/420, already at the paper level; ~4.8 min/epoch |
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

SASRec ML-1M checkpoint, `--task sequential`, batch 8192, one warm-up plus
three timed passes:

| dtype | autocast_active | s/pass | rows/s (mean) | rows/s (median) |
| --- | --- | --- | --- | --- |
| fp32 | false | 0.0266 / 0.0230 / 0.0233 | 249,935 | 259,380 |
| bf16 | **true** | 0.0249 / 0.0212 / 0.0219 | **267,840 (+7.2 %)** | 276,425 (+6.6 %) |

Training probe, `sasrec_paper` with `train.model.torch_dtype=bf16` (one card,
`ENABLE_FALLBACK_TO_CPU=0`, no CPU fallback, no error):

| dtype | epochs (best) | mean s/epoch | samples/s | train time | HR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 163 (143) | 0.725 | 8,337 | 126.9 s | 0.81738 | 0.58953 |
| bf16 | 146 (126) | 0.782 | 7,729 | 121.7 s | **0.81589** | **0.58953** |

`autocast_active` is the tool's own check (`torch.is_autocast_enabled("mlu")`).
For training the context was verified independently: a matmul inside
`autocast_context(device, "bf16", platform_mlu)` returns `torch.bfloat16`.

Reading: BF16 inference is ~7 % faster on this model.  BF16 training is
numerically indistinguishable (NDCG identical, HR@10 −0.18 %) but ~8 % slower
per epoch — a 212 k-parameter model is too small for the conversion overhead to
pay off, and it finishes earlier only because it early-stops at epoch 146
instead of 163.  Batch BF16 where the model is compute-bound (ads models,
LightGCN), not for SASRec throughput.

Probe checkpoint: `<ADREC_OUT_ROOT>/rec/ckpts/sasrec_paper-mlu` under a separate
`ADREC_OUT_ROOT` (removed after recording); the figures above are from its
`metrics.json` / `history.json` and the two bench JSON lines.

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
