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

"""Build SASRec splits from the preprocessed ``user item`` text files (Beauty, Games, Steam).

kang205/SASRec ships its datasets as text files with one ``user item`` pair per line, ordered
by timestamp (``data/Beauty.txt``, ``data/Video.txt`` for Amazon Video Games, ``data/Steam.txt``);
pmixer/SASRec.pytorch carries the same files (identical md5) under ``python/data/``.  Nothing is
filtered here, and the files match the statistics of the paper's Table II (Games 31,013 users and
23,715 items; Steam 334,730 and 13,047; Beauty 57,289 items and 0.4M actions).  The paper says
users and items with fewer than 5 actions are discarded, but the released files still contain
shorter sequences (about 11,978 users of Beauty have fewer than 5), so they are not strictly 5-core.

Protocol of Kang & McAuley (ICDM 2018), as in ``prepare_movielens_sasrec.py`` and in the official
``data_partition``:

* leave-one-out: the last item is the test target, the second to last the validation target,
  everything before is training;
* a user with fewer than 3 interactions has no validation or test item and is only used for
  training, as in the official code (the official sampler also never draws a user whose training
  sequence has a single item, so such users get no training row here);
* the target is ranked against 100 items the user never interacted with, drawn once with a
  fixed seed so every device is evaluated on identical candidate lists.

``--train-negatives`` selects which items the training sampler must not draw as negatives:
``all`` (default) uses every item of the user, as ``prepare_movielens_sasrec.py`` does;
``train`` uses only the training items, as the reference sampler (``sampler.py``) does.

    python examples/rec/prepare_sasrec_txt.py --dataset beauty --input <SASRec>/data/Beauty.txt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# name -> (output directory, default input file name)
DATASETS = {
    "beauty": ("beauty_sasrec", "Beauty.txt"),
    "games": ("games_sasrec", "Video.txt"),
    "steam": ("steam_sasrec", "Steam.txt"),
}


def left_pad(sequence: list[int], length: int) -> np.ndarray:
    row = np.zeros(length, dtype=np.int32)
    tail = sequence[-length:]
    if tail:
        row[-len(tail):] = tail
    return row


def sample_negatives(rng: np.random.Generator, seen: set[int], n_items: int, count: int) -> list[int]:
    """``count`` distinct items outside ``seen``, drawn in blocks (the item catalogue is large)."""

    chosen: list[int] = []
    taken: set[int] = set()
    while len(chosen) < count:
        for item in rng.integers(1, n_items + 1, size=2 * count).tolist():
            if item in seen or item in taken:
                continue
            taken.add(item)
            chosen.append(item)
            if len(chosen) == count:
                break
    return chosen


def main() -> None:
    root = os.environ.get("FLAGOS_REC_DATA", "/flagos-rec/datasets")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--input", default=None, help="the user-item text file "
                                                      "(default: <root>/raw/sasrec/<file>)")
    parser.add_argument("--output", default=None, help="default: <root>/<dataset>_sasrec; <root> is $FLAGOS_REC_DATA")
    parser.add_argument("--max-len", type=int, default=50, help="the paper uses 50 for these datasets (200 for ML-1M)")
    parser.add_argument("--eval-negatives", type=int, default=100)
    parser.add_argument("--train-negatives", choices=["all", "train"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    directory, filename = DATASETS[args.dataset]
    source = Path(args.input or f"{root}/raw/sasrec/{filename}")

    sequences: dict[int, list[int]] = defaultdict(list)
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            user, item = line.split()
            sequences[int(user)].append(int(item))
    users = sorted(sequences)
    item_map = {item: index + 1 for index, item in enumerate(sorted({i for s in sequences.values() for i in s}))}
    n_users, n_items = len(users), len(item_map)

    rng = np.random.default_rng(args.seed)
    length = args.max_len
    train = {k: [] for k in ("user", "seq", "pos")}
    evaluation = {split: {k: [] for k in ("user", "seq", "candidates")} for split in ("validation", "test")}
    user_items_ptr = [0]
    user_items: list[int] = []
    for index, user in enumerate(users):
        sequence = [item_map[item] for item in sequences[user]]
        has_eval = len(sequence) >= 3
        history = sequence[:-2] if has_eval else sequence
        seen = set(sequence)
        user_items.extend(sorted(seen if args.train_negatives == "all" else set(history)))
        user_items_ptr.append(len(user_items))
        if len(history) >= 2:  # at least one (input, next item) pair
            train["user"].append(index)
            train["seq"].append(left_pad(history[:-1], length))
            train["pos"].append(left_pad(history[1:], length))
        if not has_eval:
            continue
        for split, inputs, target in (
            ("validation", history, sequence[-2]),
            ("test", sequence[:-1], sequence[-1]),
        ):
            evaluation[split]["user"].append(index)
            evaluation[split]["seq"].append(left_pad(inputs, length))
            candidates = [target] + sample_negatives(rng, seen, n_items, args.eval_negatives)
            evaluation[split]["candidates"].append(np.asarray(candidates, dtype=np.int32))

    output = Path(args.output or f"{root}/{directory}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "train.npz",
        user=np.asarray(train["user"], dtype=np.int32),
        seq=np.stack(train["seq"]),
        pos=np.stack(train["pos"]),
        user_items_ptr=np.asarray(user_items_ptr, dtype=np.int64),
        user_items=np.asarray(user_items, dtype=np.int32),
    )
    for split, columns in evaluation.items():
        np.savez(
            output / f"{split}.npz",
            user=np.asarray(columns["user"], dtype=np.int32),
            seq=np.stack(columns["seq"]),
            candidates=np.stack(columns["candidates"]),
        )
    lengths = [len(sequences[user]) for user in users]
    meta = {
        "name": f"{args.dataset}-sasrec",
        "task": "sequential",
        "n_users": n_users,
        "n_items": n_items,
        "max_len": length,
        "interactions": int(sum(lengths)),
        "avg_sequence_length": float(np.mean(lengths)),
        "train_rows": len(train["user"]),
        "eval_users": len(evaluation["test"]["user"]),
        "eval_negatives": args.eval_negatives,
        "train_negatives": args.train_negatives,
        "seed": args.seed,
        "source": str(source.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
