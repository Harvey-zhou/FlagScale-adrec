# Cambricon MLU runs (ads / rec)

The ads and rec models run on Cambricon cards through the vendor stack:
`torch_mlu` + CNCL, driven by
[`flagscale/platforms/platform_mlu.py`](../flagscale/platforms/platform_mlu.py)
(`device_type=mlu`, backend `cncl`) and launched with
[`run-adrec-cambricon.sh`](run-adrec-cambricon.sh).  Torch-FL is not involved;
its Cambricon backend is a separate track.  Model, loss, data and metric code
are the ones used on CUDA.

Status: adapter and launcher written, **not yet run on a Cambricon card**.
Everything marked TODO is filled in from the cluster.

## Cluster environment

Most rows come straight from `<ADREC_OUT_ROOT>/<module>/logs/preflight.log`
(`PREFLIGHT_ONLY=1 bash examples/run-adrec-cambricon.sh ads din 0`).

| Item | Value | Source |
| --- | --- | --- |
| Cluster / host | TODO | |
| Card model and memory | TODO | `cnmon` in preflight.log |
| Driver | TODO | `cnmon` |
| NEUWARE / CNToolkit, CNNL, CNCL versions | TODO | |
| Container image | TODO | |
| Python | TODO | preflight.log |
| torch / torch_mlu | TODO | preflight.log |
| fp64 on card | TODO | preflight.log |
| Page-locked host memory | TODO | preflight.log |
| CNCL available | TODO | preflight.log |
| `ADREC_VENV` (or env activation script) | TODO | |
| Data root (`ADREC_DATA_ROOT`), md5 checked against EVIDENCE.md | TODO | |

## Order of work

1. **Preflight**: `PREFLIGHT_ONLY=1 bash examples/run-adrec-cambricon.sh ads din 0`.
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
| ads `din` | TODO | | | |
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
