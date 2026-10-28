# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-memory datasets for the recommendation example (examples/rec).

The preprocessed splits are small integer arrays written as ``.npz`` by
``examples/rec/prepare_*.py`` and are loaded fully into host memory.  Each
``__getitem__`` receives a whole list of row indices from a ``BatchSampler``
and returns one batch through fancy indexing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def load_meta(path: str | Path) -> dict[str, Any]:
    with (Path(path) / "meta.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split(path: str | Path, split: str, max_rows: int | None = None) -> dict[str, np.ndarray]:
    arrays = dict(np.load(Path(path) / f"{split}.npz"))
    if max_rows:
        # ``user_items*`` is a per-user lookup table, not a row-aligned column.
        arrays = {
            key: value if key.startswith("user_items") else value[: int(max_rows)]
            for key, value in arrays.items()
        }
    return arrays


class BatchIndexDataset(Dataset[dict[str, torch.Tensor]]):
    """Return a batch of rows for a list of indices."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.tensors = {
            key: torch.from_numpy(value.astype(np.float32 if value.dtype.kind == "f" else np.int64))
            for key, value in arrays.items()
        }
        self.rows = len(next(iter(self.tensors.values())))

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long)
        return {key: value[index] for key, value in self.tensors.items()}


class MembershipIndex:
    """Vectorized "is item in the user's interaction set" test.

    Every (user, item) pair is encoded as ``user * n_items + item``; lookups
    binary-search a sorted array of those keys.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, n_items: int) -> None:
        self.n_items = int(n_items)
        self.keys = np.unique(users.astype(np.int64) * self.n_items + items.astype(np.int64))

    def contains(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        query = users.astype(np.int64) * self.n_items + items.astype(np.int64)
        position = np.searchsorted(self.keys, query)
        position = np.minimum(position, len(self.keys) - 1)
        return self.keys[position] == query


def _worker_rng(seed: int) -> np.random.Generator:
    info = torch.utils.data.get_worker_info()
    base = info.seed if info is not None else seed + int(os.environ.get("RANK", "0"))
    return np.random.default_rng(base % (2**32))


class SequentialTrainDataset(BatchIndexDataset):
    """SASRec training rows with one fresh negative per position.

    A negative is drawn uniformly and redrawn while it belongs to the user's
    interaction set, as in the reference sampler; negatives are resampled on
    every access, so each epoch sees new ones.
    """

    def __init__(self, arrays: dict[str, np.ndarray], n_items: int, seed: int) -> None:
        ptr = arrays.pop("user_items_ptr")
        items = arrays.pop("user_items")
        super().__init__(arrays)
        self.n_items = int(n_items)
        self.seed = int(seed)
        self._rng: np.random.Generator | None = None
        users = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
        self.seen = MembershipIndex(users, items, self.n_items + 1)

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        if self._rng is None:
            self._rng = _worker_rng(self.seed)
        batch = super().__getitem__(indices)
        users = np.broadcast_to(batch["user"].numpy()[:, None], batch["pos"].shape)
        pos = batch["pos"].numpy()
        neg = self._rng.integers(1, self.n_items + 1, size=pos.shape)
        for _ in range(100):
            clash = self.seen.contains(users, neg)
            if not clash.any():
                break
            neg[clash] = self._rng.integers(1, self.n_items + 1, size=int(clash.sum()))
        neg[pos == 0] = 0
        batch["neg"] = torch.from_numpy(neg.astype(np.int64))
        return batch


class BPRSampleDataset(Dataset[dict[str, torch.Tensor]]):
    """LightGCN's per-epoch BPR triples (UniformSample_original_python).

    Each epoch draws ``n_train`` users uniformly, one positive uniformly from
    the user's training items and one negative uniformly from the items the
    user never interacted with in training.  ``resample`` is called at the
    start of every epoch.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, n_users: int, n_items: int, seed: int) -> None:
        order = np.lexsort((items, users))
        self.users_sorted = users[order].astype(np.int64)
        self.items_sorted = items[order].astype(np.int64)
        self.ptr = np.searchsorted(self.users_sorted, np.arange(n_users + 1))
        self.n_users, self.n_items = int(n_users), int(n_items)
        self.n_train = len(users)
        self.seen = MembershipIndex(users, items, n_items)
        self.rng = np.random.default_rng(seed)
        self.samples: dict[str, torch.Tensor] = {}
        self.resample()

    def resample(self) -> None:
        users = self.rng.integers(0, self.n_users, size=self.n_train)
        counts = self.ptr[users + 1] - self.ptr[users]
        keep = counts > 0
        users, counts = users[keep], counts[keep]
        pos = self.items_sorted[self.ptr[users] + (self.rng.random(len(users)) * counts).astype(np.int64)]
        neg = self.rng.integers(0, self.n_items, size=len(users))
        for _ in range(100):
            clash = self.seen.contains(users, neg)
            if not clash.any():
                break
            neg[clash] = self.rng.integers(0, self.n_items, size=int(clash.sum()))
        self.samples = {
            "user": torch.from_numpy(users),
            "pos": torch.from_numpy(pos),
            "neg": torch.from_numpy(neg.astype(np.int64)),
        }

    def __len__(self) -> int:
        return len(self.samples["user"])

    def __getitem__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long)
        return {key: value[index] for key, value in self.samples.items()}


def batch_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int = 0,
    seed: int = 0,
    distributed: bool = False,
) -> DataLoader:
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=False)
    elif shuffle:
        sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(seed))
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
        batch_size=None,
        num_workers=max(0, int(num_workers)),
        pin_memory=pin_memory,
        persistent_workers=False,
    )


@dataclass
class RecData:
    """Everything a task needs: loaders, evaluation tables and metadata."""

    meta: dict[str, Any]
    train: DataLoader
    validation: DataLoader | None = None
    test: DataLoader | None = None
    items: DataLoader | None = None
    graph: dict[str, torch.Tensor] | None = None
    train_items: MembershipIndex | None = None
    test_items: dict[str, np.ndarray] | None = None
    sampler: BPRSampleDataset | None = None


def normalized_adjacency(users: np.ndarray, items: np.ndarray, n_users: int) -> dict[str, torch.Tensor]:
    """Symmetric ``D^-1/2 A D^-1/2`` of the bipartite graph as directed edges."""

    users = users.astype(np.int64)
    items = items.astype(np.int64) + n_users
    src = np.concatenate([users, items])
    dst = np.concatenate([items, users])
    degree = np.bincount(src, minlength=int(max(src.max(), dst.max())) + 1).astype(np.float64)
    weight = 1.0 / np.sqrt(degree[src] * degree[dst])
    return {
        "src": torch.from_numpy(src),
        "dst": torch.from_numpy(dst),
        "weight": torch.from_numpy(weight.astype(np.float32)),
    }


def _random_rec_batches(num_embeddings: int, batch_size: int, num_batches: int, seed: int) -> dict[str, np.ndarray]:
    """Random (user, movie) batches drawn in the order of TorchRec's ``RandomRecDataset``.

    ``examples/retrieval`` of TorchRec feeds the two-tower model with ``RandomRecDataset``
    (keys ``userId`` / ``movieId``, ``ids_per_feature=1``, no dense features).  Per batch
    and per feature it draws a 0/1 length for every sample (a sample with length 0 has an
    empty bag, whose pooled embedding is zero), then that many ids, and finally the
    binary labels.  Using the same generator and draw order gives the same batches for
    the same seed on every device, because everything is drawn on the host.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    columns: dict[str, list[torch.Tensor]] = {
        "user_id": [], "user_len": [], "movie_id": [], "movie_len": [], "label": []}
    for _ in range(num_batches):
        for key in ("user", "movie"):
            length = torch.randint(0, 2, (batch_size,), dtype=torch.int32, generator=generator)
            values = torch.randint(0, num_embeddings, (int(length.sum()),), generator=generator)
            ids = torch.zeros(batch_size, dtype=torch.int64)
            ids[length > 0] = values
            columns[f"{key}_id"].append(ids)
            columns[f"{key}_len"].append(length.to(torch.int64))
        torch.randn(batch_size, 0, generator=generator)  # num_dense = 0 draws nothing; kept for parity
        columns["label"].append(torch.randint(0, 2, (batch_size,), generator=generator))
    return {key: torch.cat(parts).numpy() for key, parts in columns.items()}


def _build_synthetic_two_tower(data_cfg: DictConfig, batch_size: int, eval_batch_size: int,
                               pin_memory: bool, seed: int) -> RecData:
    """TorchRec's random two-tower workload: a few cached batches, cycled for N iterations.

    ``RandomRecDataset`` pre-generates ``num_generated_batches`` (default 10) batches and
    cycles through them, so a 100-iteration run sees the same 320 samples ten times.  The
    training batches are replayed in that order (no shuffling); the held-out set is drawn
    with another seed.  Nothing is read from disk.
    """

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    num_embeddings = int(data_cfg.get("num_embeddings", 1024 ** 2))
    cached = int(data_cfg.get("num_generated_batches", 10))
    iterations = int(data_cfg.get("num_iterations", 100))
    eval_batches = int(data_cfg.get("eval_batches", 10))
    pool = _random_rec_batches(num_embeddings, batch_size, cached, seed)
    order = np.concatenate([np.arange(batch_size) + (i % cached) * batch_size for i in range(iterations)])
    held_out = _random_rec_batches(num_embeddings, batch_size, eval_batches, seed + 1)
    meta = {"name": "torchrec-random-two-tower", "task": "synthetic_two_tower", "num_embeddings": num_embeddings,
            "batch_size": batch_size, "num_generated_batches": cached, "num_iterations": iterations,
            "eval_batches": eval_batches, "seed": seed}
    return RecData(
        meta=meta,
        train=batch_loader(BatchIndexDataset({key: value[order] for key, value in pool.items()}),
                           batch_size, False, pin_memory, 0, seed, distributed),
        validation=batch_loader(BatchIndexDataset(pool), eval_batch_size, False, pin_memory),
        test=batch_loader(BatchIndexDataset(held_out), eval_batch_size, False, pin_memory),
    )


def build_rec_data(task: str, data_cfg: DictConfig, batch_size: int, eval_batch_size: int,
                   pin_memory: bool, seed: int) -> RecData:
    if task == "synthetic_two_tower":
        return _build_synthetic_two_tower(data_cfg, batch_size, eval_batch_size, pin_memory, seed)
    path = str(data_cfg.path)
    meta = load_meta(path)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    workers = int(data_cfg.get("num_workers", 0) or 0)
    train_max = data_cfg.get("max_samples")
    eval_max = data_cfg.get("eval_max_samples")
    if task == "sequential":
        train = SequentialTrainDataset(load_split(path, "train", train_max), meta["n_items"], seed)
        return RecData(
            meta=meta,
            train=batch_loader(train, batch_size, True, pin_memory, workers, seed, distributed),
            validation=batch_loader(BatchIndexDataset(load_split(path, "validation", eval_max)),
                                    eval_batch_size, False, pin_memory),
            test=batch_loader(BatchIndexDataset(load_split(path, "test", eval_max)),
                              eval_batch_size, False, pin_memory),
        )
    if task == "two_tower":
        train = BatchIndexDataset(load_split(path, "train", train_max))
        return RecData(
            meta=meta,
            train=batch_loader(train, batch_size, True, pin_memory, workers, seed, distributed),
            test=batch_loader(BatchIndexDataset(load_split(path, "test", eval_max)),
                              eval_batch_size, False, pin_memory),
            items=batch_loader(BatchIndexDataset(load_split(path, "items")), eval_batch_size, False, pin_memory),
        )
    if task == "graph":
        train_pairs = load_split(path, "train")
        test_pairs = load_split(path, "test")
        sampler = BPRSampleDataset(train_pairs["user"], train_pairs["item"], meta["n_users"], meta["n_items"], seed)
        return RecData(
            meta=meta,
            train=batch_loader(sampler, batch_size, True, pin_memory, 0, seed, distributed),
            graph=normalized_adjacency(train_pairs["user"], train_pairs["item"], meta["n_users"]),
            train_items=sampler.seen,
            test_items=test_pairs,
            sampler=sampler,
        )
    raise ValueError(f"Unsupported rec task {task!r}")


__all__ = [
    "BPRSampleDataset",
    "BatchIndexDataset",
    "MembershipIndex",
    "RecData",
    "SequentialTrainDataset",
    "batch_loader",
    "build_rec_data",
    "load_meta",
    "load_split",
    "normalized_adjacency",
]
