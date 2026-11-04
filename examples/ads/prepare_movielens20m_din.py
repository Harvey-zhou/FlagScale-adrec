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

"""Encode MovieLens-20M for DIN.

The DIN paper (KDD'18) has no MovieLens experiment -- its datasets are Amazon
and Alibaba -- so this is the authors' Amazon protocol applied to MovieLens-20M
and there is no paper number to compare against:

* a user's rated movies in timestamp order are the sequence; every rating is an
  interaction (explicit feedback, the analogue of an Amazon review);
* train: for position k = 1 .. n-2 the target is item k and the history is the
  items before it, with one uniformly sampled negative (an item the user has not
  rated) per positive;
* test: the last item is the positive, with one sampled negative;
* ``cate_id`` is the movie's first genre, as in the DSSM builders;
* vocabularies are fitted on the ratings, index 0 is padding, known tokens are
  1..n, n + 1 is OOV, vocab_size = n + 2 (the FuxiCTR v2 convention used for
  Amazon);
* histories keep the **most recent** ``max_len`` items.  The Amazon builder
  keeps the first ``max_len`` because its histories average 13 items;
  MovieLens-20M averages ~145, so the first 100 would drop the recent behaviour
  the target depends on.

The column layout matches ``prepare_amazon_electronics_x1.py`` so the DIN model
and the ads loader work unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def window_view(sequence: np.ndarray, max_len: int) -> np.ndarray:
    """``view[p]`` = the ``max_len`` items before position ``p`` (left-padded to ``max_len``)."""

    padded = np.concatenate([np.zeros(max_len, dtype=np.int32), sequence])
    return np.lib.stride_tricks.sliding_window_view(padded, max_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/data2/jiangxuhui/datasets/raw/ml-20m/ml-20m")
    parser.add_argument("--output", default="/data2/jiangxuhui/datasets/movielens_20m_din")
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    raw = Path(args.raw)

    ratings = pd.read_csv(raw / "ratings.csv", usecols=["userId", "movieId", "timestamp"],
                          dtype={"userId": np.int32, "movieId": np.int32, "timestamp": np.int64})
    movies = pd.read_csv(raw / "movies.csv", usecols=["movieId", "genres"], dtype={"movieId": np.int32})
    movies["cate"] = movies["genres"].str.split("|").str[0]
    genre_of = dict(zip(movies["movieId"].tolist(), movies["cate"].tolist()))
    print(f"ratings {len(ratings)}, movies {len(movies)}", flush=True)

    order = np.lexsort((ratings["timestamp"].to_numpy(), ratings["userId"].to_numpy()))
    users = ratings["userId"].to_numpy()[order]
    items = ratings["movieId"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    uniq_users = users[starts]

    item_tokens, item_counts = np.unique(items, return_counts=True)
    rank = np.lexsort((item_tokens, -item_counts))
    item_ids = np.empty(len(item_tokens), dtype=np.int64)
    item_ids[rank] = np.arange(1, len(item_tokens) + 1)
    vocab_item = dict(zip(item_tokens.tolist(), item_ids.tolist()))
    cate_tokens, cate_counts = np.unique(movies["cate"].to_numpy(), return_counts=True)
    cate_rank = np.lexsort((cate_tokens, -cate_counts))
    cate_ids = np.empty(len(cate_tokens), dtype=np.int64)
    cate_ids[cate_rank] = np.arange(1, len(cate_tokens) + 1)
    vocab_cate = dict(zip(cate_tokens.tolist(), cate_ids.tolist()))
    vocab_sizes = {"item_id": len(vocab_item) + 2, "cate_id": len(vocab_cate) + 2}

    item_code = np.array([vocab_item[i] for i in items.tolist()], dtype=np.int32)
    cate_code = np.array([vocab_cate[genre_of[i]] for i in items.tolist()], dtype=np.int32)
    n_item_tokens = len(item_tokens)

    def sample_negatives(seq: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        drawn = rng.integers(1, n_item_tokens + 1, size=count).astype(np.int32)
        if len(seq):
            for _ in range(16):
                collision = np.isin(drawn, seq)
                if not collision.any():
                    break
                drawn[collision] = rng.integers(1, n_item_tokens + 1, size=int(collision.sum())).astype(np.int32)
        return drawn

    rng = np.random.default_rng(args.seed)
    train_blocks: list[tuple] = []
    test_blocks: list[tuple] = []
    for index in range(len(uniq_users)):
        begin, end = int(starts[index]), int(ends[index])
        length = end - begin
        if length < 3:
            continue
        seq_item = item_code[begin:end]
        seq_cate = cate_code[begin:end]
        user = int(uniq_users[index])

        item_windows = window_view(seq_item, args.max_len)
        cate_windows = window_view(seq_cate, args.max_len)

        # --- train: positions 1 .. n-2
        targets = seq_item[1:length - 1]
        hist_item = item_windows[1 : length - 1]
        hist_cate = cate_windows[1 : length - 1]
        negatives = sample_negatives(seq_item, len(targets), rng)
        train_blocks.append((user, targets, negatives, hist_item, hist_cate))

        # --- test: the last item, history = every item before it
        test_blocks.append((user, seq_item[length - 1 : length], sample_negatives(seq_item, 1, rng),
                            item_windows[length - 1 : length], cate_windows[length - 1 : length]))

    def assemble(blocks: list[tuple]) -> dict[str, np.ndarray]:
        counts = np.array([len(block[1]) for block in blocks], dtype=np.int64)
        total = int(counts.sum())
        user_col = np.repeat(np.array([block[0] for block in blocks], dtype=np.int64), 2 * counts)
        pos_item = np.concatenate([np.asarray(block[1]) for block in blocks])
        neg_item = np.concatenate([np.asarray(block[2]) for block in blocks])
        hist_item = np.concatenate([np.asarray(block[3]) for block in blocks])
        hist_cate = np.concatenate([np.asarray(block[4]) for block in blocks])
        item_col = np.empty(2 * total, dtype=np.int32)
        item_col[0::2] = pos_item
        item_col[1::2] = neg_item
        cate_col = code_to_cate[item_col]
        out = {
            "user_id": user_col,
            "item_id": item_col,
            "cate_id": cate_col,
            "label": np.tile(np.array([1, 0], dtype=np.int32), total),
            "item_history": np.repeat(hist_item, 2, axis=0),
            "cate_history": np.repeat(hist_cate, 2, axis=0),
        }
        return out

    # code -> genre code lookup for targets and negatives
    code_to_cate = np.zeros(n_item_tokens + 2, dtype=np.int32)
    for token in item_tokens.tolist():
        code_to_cate[vocab_item[token]] = vocab_cate[genre_of[token]]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"train": assemble(train_blocks), "test": assemble(test_blocks)}
    for split, columns in arrays.items():
        np.savez(output / f"{split}.npz", **columns)
        print(split, {k: v.shape for k, v in columns.items()}, flush=True)
    meta = {
        "name": "movielens_20m_din",
        "vocab_sizes": vocab_sizes,
        "rows": {split: int(len(columns["label"])) for split, columns in arrays.items()},
        "positive_ratio": {split: float(columns["label"].mean()) for split, columns in arrays.items()},
        "max_len": args.max_len,
        "protocol": "DIN authors' Amazon protocol applied to ML-20M (the paper has no MovieLens experiment)",
        "deviations": "history keeps the most recent max_len items; one uniform negative per positive",
        "seed": args.seed,
        "source": str(raw.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "vocab_sizes"}, indent=2))


if __name__ == "__main__":
    main()
