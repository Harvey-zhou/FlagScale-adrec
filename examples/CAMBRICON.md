# Cambricon MLU runs (ads / rec)

The ads and rec models run on Cambricon cards through the vendor stack:
`torch_mlu` + CNCL, driven by
[`flagscale/platforms/platform_mlu.py`](../flagscale/platforms/platform_mlu.py)
(`device_type=mlu`, backend `cncl`) and launched with
[`run-adrec-cambricon.sh`](run-adrec-cambricon.sh).  Torch-FL is not involved;
its Cambricon backend is a separate track.  Model, loss, data and metric code
are the ones used on CUDA.

Status: adapter, launcher and container wrapper written.  First card run done
on 2026-11-03: preflight plus the `ads`/`din` smoke on one MLU590-M9.  The
`rec` configs and the multi-card runs are still TODO.

## Cluster environment

Most rows come straight from `<ADREC_OUT_ROOT>/<module>/logs/preflight.log`
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
| Page-locked host memory | yes (`FLAGSCALE_PIN_MEMORY=true` may be used) | preflight.log |
| CNCL available | yes (`torch.distributed.is_cncl_available()`) | preflight.log |
| `ADREC_VENV` (or env activation script) | `/torch/venv3/pytorch` inside the 26.08 container (the login shell already activates it) | preflight.log |
| Data root (`ADREC_DATA_ROOT`), md5 checked against EVIDENCE.md | `/srv/jiangxuhui/datasets`; `AmazonElectronics_x1` raw md5 `e9bf80b9…` / `57a20e82…` match EVIDENCE.md, encoded with `prepare_amazon_electronics_x1.py` | md5sum, prepare output |

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
| Internet / mirrors | `huggingface.co` times out; `hf-mirror.com` works (`HF_ENDPOINT=https://hf-mirror.com`) |
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
3. **Full single-card runs**, compared with the CUDA baseline and the reference
   numbers in `ads/EVIDENCE.md` / `rec/EVIDENCE.md`.
4. **Two cards** (DDP over CNCL), e.g. `bash examples/run-adrec-cambricon.sh ads deepfm 0,1`.

## Operator probe results

| Config | Strict smoke | Ops on CPU with fallback on | Ops with no MLU kernel | Notes |
| --- | --- | --- | --- | --- |
| ads `din` | yes (exit 0, fallback off) | none | none | CNNL warns `cnnlMasked_v4` is deprecated; 3-step eval AUC 0.5667 (not a result) |
| ads `deepfm` | TODO | | | |
| ads `dcn_mix` | TODO | | | |
| rec `sasrec` | TODO | | | |
| rec `two_tower` | TODO | | | |
| rec `lightgcn` | TODO | | | `sparse` vs `index_add` |

## Training results

| Config | Cards | Test metric | CUDA baseline | Reference | Train samples/s | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| TODO | | | | | | |

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
* **Page-locked memory off by default** until the preflight shows it works;
  `FLAGSCALE_PIN_MEMORY=true` turns it on.

Not yet verified on the cluster: `MLU_VISIBLE_DEVICES` as the card selector,
`torch.distributed.is_cncl_available()`, `torch.mlu.mem_get_info` and the
fallback variable's name in the installed torch_mlu version.
