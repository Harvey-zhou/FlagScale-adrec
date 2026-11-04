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

"""Build a DSSM two-tower MovieLens-20M dataset.

Same recipe as ``prepare_movielens_dssm.py`` (torch-rechub's
``generate_seq_feature_match`` / ``gen_model_input``), with one structural
difference forced by the data: **MovieLens-20M has no user demographics**, so
the user tower sees ``user_id`` plus the mean-pooled history only (no gender,
age, occupation or zip).  Everything else is unchanged:

* ``cate_id`` is the first genre;
* ids are label-encoded to 1..n in sorted order (vocabulary size n + 1);
* interactions are ordered by timestamp; for each user and each position
  ``i >= 1`` the history is the items before ``i``; the last position is the
  test sample, every other position is a positive training sample followed by
  ``neg_ratio`` negatives;
* negatives are drawn with probability proportional to ``count ** 0.75``
  (``sample_method=1``) from one pre-generated list consumed in order;
* histories are left-padded and truncated to the most recent ``seq_max_len``
  items.

Rows are built per user in blocks: MovieLens-20M produces ~80 M training rows,
which the list-of-tuples approach of the 1 M builder would not hold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def window_view(sequence: np.ndarray, length: int) -> np.ndarray:
    """``view[p]`` = the ``length`` items before position ``p``, left-padded."""

    padded = np.concatenate([np.zeros(length, dtype=np.int32), sequence])
    return np.lib.stride_tricks.sliding_window_view(padded, length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/data2/jiangxuhui/datasets/raw/ml-20m/ml-20m")
    parser.add_argument("--output", default="/data2/jiangxuhui/datasets/rec/movielens_20m_dssm")
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--seq-max-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    raw = Path(args.raw)

    ratings = pd.read_csv(raw / "ratings.csv", usecols=["userId", "movieId", "timestamp"],
                          dtype={"userId": np.int32, "movieId": np.int32, "timestamp": np.int64})
    movies = pd.read_csv(raw / "movies.csv", usecols=["movieId", "genres"], dtype={"movieId": np.int32})
    movies["cate"] = movies["genres"].str.split("|").str[0]
    print(f"ratings {len(ratings)}, movies {len(movies)}", flush=True)

    order = np.lexsort((ratings["timestamp"].to_numpy(), ratings["userId"].to_numpy()))
    users = ratings["userId"].to_numpy()[order]
    items = ratings["movieId"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    uniq_users = users[starts]

    user_tokens = np.unique(users)
    user_code = np.searchsorted(user_tokens, users).astype(np.int32) + 1
    movie_tokens = np.unique(items)
    movie_code = np.searchsorted(movie_tokens, items).astype(np.int32) + 1

    genre_of = dict(zip(movies["movieId"].tolist(), movies["cate"].tolist()))
    cate_tokens = np.unique(movies["cate"].to_numpy())
    cate_lookup = {str(token): index + 1 for index, token in enumerate(cate_tokens.tolist())}
    movie_cate = np.array([cate_lookup[genre_of[int(m)]] for m in movie_tokens.tolist()], dtype=np.int32)

    counts = np.bincount(movie_code, minlength=len(movie_tokens) + 1)[1:]
    probabilities = counts.astype(np.float64) ** 0.75
    probabilities /= probabilities.sum()

    lengths = ends - starts
    train_total = int((np.maximum(lengths - 2, 0) * (1 + args.neg_ratio)).sum())
    test_total = int((lengths >= 3).sum())
    print(f"train rows {train_total}, test users {test_total}", flush=True)

    train_user = np.empty(train_total, dtype=np.int32)
    train_movie = np.empty(train_total, dtype=np.int32)
    train_label = np.empty(train_total, dtype=np.int32)
    train_hist = np.zeros((train_total, args.seq_max_len), dtype=np.int32)
    test_user = np.empty(test_total, dtype=np.int32)
    test_movie = np.empty(test_total, dtype=np.int32)
    test_hist = np.zeros((test_total, args.seq_max_len), dtype=np.int32)

    rng = np.random.default_rng(args.seed)
    negatives_per_row = train_total // (1 + args.neg_ratio)
    pool = np.searchsorted(movie_tokens, rng.choice(movie_tokens, size=negatives_per_row * args.neg_ratio,
                                                    replace=True, p=probabilities)).astype(np.int32) + 1

    cursor = 0
    position = 0
    test_position = 0
    stride = 1 + args.neg_ratio
    for index in range(len(uniq_users)):
        begin, end = int(starts[index]), int(ends[index])
        length = end - begin
        if length < 3:
            continue
        seq = movie_code[begin:end]
        view = window_view(seq, args.seq_max_len)
        rows = (length - 2) * stride
        block = slice(position, position + rows)

        train_user[block] = int(user_code[begin])
        block_movie = np.repeat(seq[1:length - 1], stride)
        block_label = np.tile(np.array([1] + [0] * args.neg_ratio, dtype=np.int32), length - 2)
        negative_rows = np.arange(position + 1, position + rows, stride)
        block_movie[np.arange(1, rows, stride)] = pool[cursor:cursor + len(negative_rows)]
        cursor += len(negative_rows)
        train_movie[block] = block_movie
        train_label[block] = block_label
        train_hist[block] = np.repeat(view[1:length - 1], stride, axis=0)
        position += rows

        test_user[test_position] = int(user_code[begin])
        test_movie[test_position] = seq[length - 1]
        test_hist[test_position] = view[length - 1]
        test_position += 1

    train_total, test_total = position, test_position
    shuffle = rng.permutation(train_total)
    arrays = {
        "train": {
            "user_id": train_user[:train_total][shuffle],
            "movie_id": train_movie[:train_total][shuffle],
            "cate_id": movie_cate[train_movie[:train_total][shuffle] - 1],
            "hist_movie_id": train_hist[:train_total][shuffle],
            "label": train_label[:train_total][shuffle],
        },
        "test": {
            "user_id": test_user[:test_total],
            "movie_id": test_movie[:test_total],
            "cate_id": movie_cate[test_movie[:test_total] - 1],
            "hist_movie_id": test_hist[:test_total],
        },
        "items": {"movie_id": np.arange(1, len(movie_tokens) + 1, dtype=np.int32), "cate_id": movie_cate},
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split, columns in arrays.items():
        np.savez(output / f"{split}.npz", **columns)
        print(split, {k: v.shape for k, v in columns.items()}, flush=True)

    meta = {
        "name": "movielens-20m-torch-rechub-match",
        "task": "two_tower",
        "user_fields": ["user_id"],
        "vocab_sizes": {"user_id": len(user_tokens) + 1, "movie_id": len(movie_tokens) + 1,
                        "cate_id": len(cate_tokens) + 1},
        "n_users": int(len(uniq_users)),
        "n_items": int(len(movie_tokens)),
        "interactions": int(len(ratings)),
        "train_rows": int(train_total),
        "test_rows": int(test_total),
        "neg_ratio": args.neg_ratio,
        "seq_max_len": args.seq_max_len,
        "seed": args.seed,
        "deviations": "no user demographics in MovieLens-20M: user tower = user_id + mean-pooled history",
        "source": str(raw.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
