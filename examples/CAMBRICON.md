# Cambricon MLU runs (ads / rec)

The ads and rec models run on Cambricon cards through the vendor stack:
`torch_mlu` + CNCL, driven by
[`flagscale/platforms/platform_mlu.py`](../flagscale/platforms/platform_mlu.py)
(`device_type=mlu`, backend `cncl`) and launched with
[`run-adrec-cambricon.sh`](run-adrec-cambricon.sh).  Torch-FL is not involved;
its Cambricon backend is a separate track.  Model, loss, data and metric code
are the ones used on CUDA.

Status: adapter, launcher and container wrapper written.  Done on 2026-11-03 /
11-06 / 11-07: preflight, the `ads` smokes, single-card `rec` training
(`sasrec`, `dssm`, `two_tower`), the four paper-protocol SASRec runs (ML-1M,
Beauty, Games, Steam), 2-card CNCL DDP on ML-1M, and the full LightGCN
Gowalla paper run (1,000 epochs, 2 cards).  Multi-node is still TODO.

## Cluster environment

Most rows come straight from the preflight log,
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

## Host inventory

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
| MLU clock | 1200 MHz current vs 1600 MHz default — all cards capped (`Clock Setting: Active`) |
| Driver / firmware | 6.5.52 (runfile install, DKMS-built), firmware v1.5.0, `mlumsgd` running |
| Storage | `/` 437 GB (109 GB free), `/srv` 11 TB (5.9 TB free) |
| Network | 11 mlx5 HCAs, RoCE interfaces up |
| Internet / mirrors | `huggingface.co` times out; `hf-mirror.com` works (`HF_ENDPOINT=https://hf-mirror.com`); GitHub `raw.githubusercontent.com` is blocked but `api.github.com` works (contents API with `Accept: application/vnd.github.raw` supports Range) |
| Docker | installed; the login user is in the `docker` group (takes effect on a new login; `sg docker -c …` works in a running shell), set `DOCKER=docker` to skip sudo |
| Vendor bundle | shared storage: drivers 6.2.29 and 6.5.52, `.deb` packages, cp312/pt2.12.1 wheels, five container tars, SDK docs |
| Open items | clock cap above; stale `cambricon-mlu-driver-dkms 6.2.29` dpkg/dkms record; no MagicMind (inference only, not needed for training) |

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

| Config | Strict smoke | Ops on CPU with fallback on | Ops with no MLU kernel | Notes |
| --- | --- | --- | --- | --- |
| ads `din` | yes (exit 0, fallback off) | none | none | CNNL warns `cnnlMasked_v4` is deprecated; 3-step eval AUC 0.5667 (not a result) |
| ads `deepfm` | yes (exit 0, fallback off) | none | none | Criteo_x1 33.0M/8.25M/4.59M; 23.4M parameters |
| ads `dcn_mix` | yes (exit 0, fallback off) | none | none | Criteo_x1; 21.9M parameters |
| rec `sasrec` | yes (exit 0, fallback off) | none | none | ML-1M; 211,950 parameters |
| rec `dssm` | yes (exit 0, fallback off) | none | none | MovieLens-1M; 329,222 parameters; the two-tower model of the six (torch-rechub); `two_tower` is TorchRec's synthetic-data example and optional |
| rec `two_tower` | yes (exit 0, fallback off) | none | none | synthetic batches, 134.3M parameters |
| rec `lightgcn` | yes (exit 0, fallback off) | none | none | `sparse` Coo runs on MLU; `index_add` was not needed |

## Training results

| Config | Cards | Test metric | P800 same config | Reference | Train samples/s | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| rec `sasrec` (ML-1M, code defaults) | 1 | HR@10 0.8180 / NDCG@10 0.5892 | — (P800 ran `sasrec_paper`: 0.8204 / 0.5910) | 0.8190–0.8225 / 0.5897–0.5918 (pmixer code) | 8,592 | 200 epochs, 141 s total |
| rec `sasrec_paper` (ML-1M) | 1 | HR@10 0.8174 / NDCG@10 0.5895 | 0.8204 / 0.5910 (197 s) | 0.8245 / 0.5905 (paper) | 8,337 | early stop @163; 127 s |
| rec `sasrec` (ML-1M, 2-card DDP) | 2 | HR@10 0.8190 / NDCG@10 0.5925 | — | 0.8190–0.8225 / 0.5897–0.5918 (pmixer code) | 9,805 | CNCL over MLU-Link; 1.14× the 1-card throughput; 123 s |
| rec `sasrec_beauty_paper` | 1 | HR@10 0.4684 / NDCG@10 0.3092 | 0.4604 / 0.3034 (2,079 s) | 0.4854 / 0.3219 (paper) | 10,838 | early stop @381; 2,058 s |
| rec `sasrec_games_paper` | 1 | HR@10 0.7266 / NDCG@10 0.5222 | 0.7340 / 0.5281 (914 s) | 0.7410 / 0.5360 (paper) | 9,305 | early stop @133; 409 s |
| rec `sasrec_steam_paper` | 1 | HR@10 0.8614 / NDCG@10 0.6167 | 0.8589 / 0.6150 (4,702 s) | 0.8729 / 0.6306 (paper) | 10,672 | early stop @108; 3,416 s |
| rec `lightgcn_paper` (Gowalla, 2-card DDP) | 2 | recall@20 0.1820 / ndcg@20 0.1550 | 0.1820 / 0.1553 (70 h, 1 card) | 0.1823 / 0.1555 (paper) | 63,700 | 1,000 epochs, 3 h 43 m; batch 512/rank (global 1024); `index_add` |
| rec `dssm` (ML-1M) | 1 | Recall@100 0.2123 | — | 0.1474 (torch-rechub README) | 312,074 | exact full-catalogue ranking; the reference uses Annoy, so not strictly like-for-like |
| rec `two_tower` (synthetic) | 1 | neg_loss −0.6923 | — | none published | 7,045 | 100 steps; pipeline check only |

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

Throughput: the cards run capped at 1200 MHz (default 1600, see Host
inventory), so record the clock with every samples/s figure.

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
