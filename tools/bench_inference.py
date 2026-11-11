#!/usr/bin/env python3
"""Inference throughput for the efficiency table.

Definition used by the project table:

* final checkpoint;
* one untimed warm-up pass, then three timed full test passes;
* throughput = test rows / mean(synchronised H2D + model forward + D2H);
* ranking and metric computation (top-k, sorting, AUC / recall / NDCG) are
  excluded.

``--task two_tower`` (DSSM, full-catalogue retrieval) reports two throughputs
per pass, both in test queries per second: ``rows_per_s_*`` times the whole pass
(item tower over the full catalogue + user tower over the test queries + the
query x catalogue score matrix), ``online_rows_per_s_*`` leaves the catalogue
encoding out (the item vectors are built offline in a serving system).  Its
passes are short, so ``--repeat`` defaults to 10 there (3 for the other tasks)
and the median is reported next to the mean.  ``--task synthetic_two_tower``
(TorchRec's random-data example) scores (query, candidate) pairs like an ads
model and needs no ``--data``.

The model and its metadata are read from the checkpoint's ``config.json``; only
the dataset path and batch size are passed on the command line.

Examples::

    python tools/bench_inference.py --task ads --checkpoint <dir> \
        --data /data2/jiangxuhui/datasets/amazonelectronics_x1
    python tools/bench_inference.py --task sequential --checkpoint <dir> \
        --data /data2/jiangxuhui/datasets/rec/movielens_1m_sasrec
    python tools/bench_inference.py --task graph --checkpoint <dir> \
        --data /data2/jiangxuhui/datasets/rec/gowalla_lightgcn
    python tools/bench_inference.py --task two_tower --checkpoint <dir> \
        --data /flagos-rec/datasets/movielens_1m_dssm
    python tools/bench_inference.py --task synthetic_two_tower --checkpoint <dir>

Add ``--dtype bf16`` to run the forward pass under ``torch.autocast`` (the
capability probe for BF16 inference).  If autocast cannot be entered on the
device the run stops with an error instead of timing an fp32 pass.

Devices: ``--device cuda`` (default; also vendor torch builds that report their
card as ``cuda``) and ``cpu`` need nothing else.  Any other value is a FlagScale
platform name, optionally with a card index, e.g. ``--device mlu:1``; it is
resolved through ``flagscale.platforms`` (the same adapters the trainers use), and
the timing synchronises that device's own runtime, not ``torch.cuda``.  On a
platform where an operator without a device kernel can silently run on the CPU the
timing would be wrong; for ``mlu`` this tool sets ``ENABLE_FALLBACK_TO_CPU=0`` when
it is unset, so such an operator fails the run.  The result records the device and
whether autocast was active.

    env PYTHONPATH=$PWD python tools/bench_inference.py --device mlu:0 --task ads \
        --checkpoint <dir> --data <dataset> [--dtype bf16]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flagscale.models.ads import build_ads_model
from flagscale.models.rec import build_rec_model
from flagscale.train.datasets.ads import build_ads_loaders
from flagscale.train.datasets.rec import build_rec_data

NATIVE_DEVICES = ("cpu", "cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["ads", "sequential", "graph", "two_tower", "synthetic_two_tower"],
                        required=True)
    parser.add_argument("--checkpoint", required=True, help="directory with model.pt and config.json")
    parser.add_argument("--data", help="dataset directory (not needed for synthetic_two_tower)")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up passes")
    parser.add_argument("--repeat", type=int, default=None,
                        help="timed passes (default 10 for two_tower, 3 otherwise)")
    args = parser.parse_args()
    if args.task != "synthetic_two_tower" and not args.data:
        parser.error(f"--data is required for --task {args.task}")
    if args.repeat is None:
        args.repeat = 10 if args.task == "two_tower" else 3
    return args


def resolve_device(spec: str) -> tuple[torch.device, str]:
    """Return the device and the device type name autocast needs.

    ``cpu`` / ``cuda[:i]`` map straight to ``torch.device``; anything else is a
    FlagScale platform name (``mlu``, ``mlu:1``, ...) resolved through the platform
    adapter, which also selects the card.
    """

    name, _, index = spec.partition(":")
    if name in NATIVE_DEVICES:
        device = torch.device(spec)
        return device, device.type
    if name == "mlu":
        os.environ.setdefault("ENABLE_FALLBACK_TO_CPU", "0")  # read when the vendor runtime loads
    os.environ["FS_PLATFORM"] = name
    from flagscale.platforms import get_platform

    platform = get_platform()
    if platform.name() != name or not platform.is_available():
        raise RuntimeError(f"platform {name!r} is not available on this host")
    card = int(index) if index else 0
    platform.set_device(card)
    return platform.device(card), platform.amp_device_type()


def synchronize(device: torch.device) -> None:
    """Wait for the device's queued work, whichever runtime owns it."""

    if device.type == "cpu":
        return
    runtime = getattr(torch, device.type, None)
    if runtime is not None and hasattr(runtime, "synchronize"):
        runtime.synchronize()
    else:
        torch.zeros(1, device=device).cpu()  # a D2H copy waits for the stream


def autocast(device: torch.device, autocast_type: str, dtype: str):
    if dtype == "fp32" or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=autocast_type, dtype=torch.bfloat16)


def autocast_active(device: torch.device, autocast_type: str, dtype: str) -> bool:
    """True when ``--dtype bf16`` really puts the forward pass under autocast.

    PyTorch may accept an autocast request for a backend and quietly disable it, which
    would time an fp32 pass under a bf16 label; that has to stop the run.
    """

    if dtype == "fp32" or device.type == "cpu":
        return False
    with autocast(device, autocast_type, dtype):
        if not torch.is_autocast_enabled(autocast_type):
            raise RuntimeError(f"bf16 autocast is not active on {device}; refusing to time an fp32 pass as bf16")
    return True


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint / "model.pt", map_location="cpu")
    config = json.loads((checkpoint / "config.json").read_text())
    device, autocast_type = resolve_device(args.device)
    model_cfg = OmegaConf.create(config["model"])
    meta = config["meta"]

    if args.task == "ads":
        data_cfg = OmegaConf.create({"path": args.data, "valid_split": "test", "group_key": None,
                                     "num_workers": 0, "max_samples": None, "eval_max_samples": None})
        model = build_ads_model(str(model_cfg.model_type), model_cfg, meta, device)
        loader = build_ads_loaders(data_cfg, args.batch_size, args.batch_size, pin_memory=False, seed=0)[2]
    else:
        data_cfg = OmegaConf.create({"path": args.data, "num_workers": 0, "max_samples": None,
                                     "eval_max_samples": None})
        if args.task == "synthetic_two_tower":
            # No files: the held-out batches are drawn from the seed, over the checkpoint's id range.
            data_cfg = OmegaConf.create({"num_embeddings": meta["num_embeddings"],
                                         "eval_batches": meta.get("eval_batches", 10),
                                         "num_iterations": 1})
        model = build_rec_model(str(model_cfg.model_type), model_cfg, meta, device)
        data = build_rec_data(args.task, data_cfg, args.batch_size, args.batch_size, pin_memory=False, seed=0)
        loader = data.test
        if args.task == "graph":
            model.set_graph(data.graph["src"], data.graph["dst"], data.graph["weight"])
    model.load_state_dict(state)
    model.eval()
    bf16_active = autocast_active(device, autocast_type, args.dtype)

    def sync() -> None:
        synchronize(device)

    def one_pass() -> tuple[int, float, float]:
        """Return (rows, seconds, online seconds); online differs only for two_tower."""
        rows = 0
        online_started = None
        with torch.no_grad(), autocast(device, autocast_type, args.dtype):
            sync()
            started = time.perf_counter()
            if args.task == "two_tower":
                # Catalogue encoding (item tower over every item), then the queries.
                # Scores stay on device; top-k and the metrics are excluded.
                item_vectors = [model.item_embedding(to_device(batch, device)) for batch in data.items]
                items = torch.cat(item_vectors)
                sync()
                online_started = time.perf_counter()
                for batch in loader:
                    rows += int(next(iter(batch.values())).shape[0])
                    users = model.user_embedding(to_device(batch, device))      # H2D + user tower
                    users @ items.T
            elif args.task == "graph":
                # The full-ranking score matrix is model compute; top-k and the
                # metric stay excluded, so only the propagation + dot product
                # are timed and each test user counts as one row.
                users_emb, items_emb = model.all_scores()
                test_users = np.unique(data.test_items["user"].astype(np.int64))
                for begin in range(0, len(test_users), args.batch_size):
                    users = torch.as_tensor(test_users[begin:begin + args.batch_size], device=device)
                    users_emb[users] @ items_emb.T
                rows = len(test_users)
            else:
                for batch in loader:
                    rows += int(next(iter(batch.values())).shape[0])
                    moved = to_device(batch, device)                 # H2D
                    if args.task == "sequential":
                        out = model.predict(moved)
                    else:
                        out = model(moved)
                    out.detach().to("cpu", non_blocking=False)       # D2H (synchronising)
            sync()
            finished = time.perf_counter()
            return rows, finished - started, finished - (online_started or started)

    for _ in range(args.warmup):
        one_pass()

    passes = []
    for _ in range(args.repeat):
        passes.append(one_pass())
    rates = [rows / seconds for rows, seconds, _ in passes]

    result = {
        "task": args.task, "model": str(model_cfg.model_type), "dataset": meta.get("name"),
        "device": str(device), "dtype": args.dtype, "autocast_active": bf16_active, "batch_size": args.batch_size,
        "rows_per_pass": passes[0][0],
        "pass_seconds": [round(seconds, 4) for _, seconds, _ in passes],
        "rows_per_s_runs": [round(rate, 1) for rate in rates],
        "rows_per_s_mean": round(statistics.mean(rates), 1),
        "rows_per_s_median": round(statistics.median(rates), 1),
    }
    if args.task == "two_tower":
        online = [rows / online_seconds for rows, _, online_seconds in passes]
        result.update({
            "catalogue_items": sum(int(next(iter(batch.values())).shape[0]) for batch in data.items),
            "online_pass_seconds": [round(online_seconds, 4) for _, _, online_seconds in passes],
            "online_rows_per_s_runs": [round(rate, 1) for rate in online],
            "online_rows_per_s_mean": round(statistics.mean(online), 1),
            "online_rows_per_s_median": round(statistics.median(online), 1),
        })
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
