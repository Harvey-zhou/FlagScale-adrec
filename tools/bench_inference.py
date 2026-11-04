#!/usr/bin/env python3
"""Inference throughput for the efficiency table.

Definition used by the project table:

* final checkpoint;
* one untimed warm-up pass, then three timed full test passes;
* throughput = test rows / mean(synchronised H2D + model forward + D2H);
* ranking and metric computation (top-k, sorting, AUC / recall / NDCG) are
  excluded.

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
        --data /data2/jiangxuhui/datasets/rec/movielens_20m_dssm

Add ``--dtype bf16`` to run the forward pass under ``torch.autocast`` (the
capability probe for BF16 inference).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["ads", "sequential", "graph", "two_tower"], required=True)
    parser.add_argument("--checkpoint", required=True, help="directory with model.pt and config.json")
    parser.add_argument("--data", required=True, help="dataset directory")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warm-up passes")
    parser.add_argument("--repeat", type=int, default=3, help="timed passes")
    return parser.parse_args()


def autocast(device: torch.device, dtype: str):
    if dtype == "fp32" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint / "model.pt", map_location="cpu")
    config = json.loads((checkpoint / "config.json").read_text())
    device = torch.device(args.device)
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
        model = build_rec_model(str(model_cfg.model_type), model_cfg, meta, device)
        data = build_rec_data(args.task, data_cfg, args.batch_size, args.batch_size, pin_memory=False, seed=0)
        loader = data.test
        if args.task == "graph":
            model.set_graph(data.graph["src"], data.graph["dst"], data.graph["weight"])
    model.load_state_dict(state)
    model.eval()

    # Two-tower retrieval: the item index is precomputed once (it is the served
    # catalogue, not per-request compute), then each timed pass streams user
    # batches through the user tower and scores them against that index.
    item_index = None
    if args.task == "two_tower":
        vectors = []
        with torch.no_grad(), autocast(device, args.dtype):
            for batch in data.items:
                vectors.append(model.item_embedding(to_device(batch, device)))
        item_index = torch.cat(vectors)
        loader = data.test

    def one_pass() -> tuple[int, float]:
        rows = 0
        with torch.no_grad(), autocast(device, args.dtype):
            torch.cuda.synchronize() if device.type == "cuda" else None
            started = time.perf_counter()
            if args.task == "two_tower":
                for batch in loader:
                    rows += int(next(iter(batch.values())).shape[0])
                    moved = to_device(batch, device)                 # H2D
                    users = model.user_embedding(moved)
                    (users @ item_index.T).detach().to("cpu", non_blocking=False)  # D2H
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
            if device.type == "cuda":
                torch.cuda.synchronize()
            return rows, time.perf_counter() - started

    for _ in range(args.warmup):
        one_pass()

    passes = []
    for _ in range(args.repeat):
        rows, seconds = one_pass()
        passes.append((rows, seconds))
    rates = [rows / seconds for rows, seconds in passes]

    print(json.dumps({
        "task": args.task, "model": str(model_cfg.model_type), "dataset": meta.get("name"),
        "device": str(device), "dtype": args.dtype, "batch_size": args.batch_size,
        "rows_per_pass": passes[0][0],
        "pass_seconds": [round(seconds, 4) for _, seconds in passes],
        "rows_per_s_runs": [round(rate, 1) for rate in rates],
        "rows_per_s_mean": round(statistics.mean(rates), 1),
        "rows_per_s_median": round(statistics.median(rates), 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
