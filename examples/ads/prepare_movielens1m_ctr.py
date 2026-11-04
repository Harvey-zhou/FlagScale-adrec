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

"""Encode MovieLens-1M as a CTR table for DCN-Mix.

Protocol: DCN-V2 (WWW 2021) §7.1.1 -- "MovieLen-1M ... one million samples of
[user-features, movie-features, rating] triplets. The task was treated as a
regression problem. Ratings of 1 and 2 were set to 0 and 4 and 5 were set to 1.
3s were removed."  The paper also states 7 features, embedding size 30 and
batch 128 for this dataset (§7.1.3, Table 3), and a random 80/10/10 split.

Fields (all categorical):
    user_id, gender, age, occupation, zip  -- users.dat
    movie_id, cate_id (first genre)        -- movies.dat, as in the DSSM builder

Rows are ratings, so the same user appears once per rated movie.  There is no
numeric field, so ``dense`` is an empty (N, 0) array and ``num_numeric`` is 0.

Encoding follows ``prepare_criteo_x1.py``: vocabularies are fitted on train,
ordered by (-count, token), index 0 is OOV, known tokens are 1..n and
vocab_size = n + 1.

The paper does not publish its split seed; this uses seed 2020 for the
80/10/10 permutation, which is recorded in ``meta.json`` as a deviation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["user_id", "gender", "age", "occupation", "zip", "movie_id", "cate_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="/data2/jiangxuhui/datasets/raw/movielens/ml-1m")
    parser.add_argument("--output", default="/data2/jiangxuhui/datasets/movielens_1m_ctr")
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    raw = Path(args.raw)

    users = pd.read_csv(raw / "users.dat", sep="::", header=None, engine="python", encoding="ISO-8859-1",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    ratings = pd.read_csv(raw / "ratings.dat", sep="::", header=None, engine="python", encoding="ISO-8859-1",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    movies = pd.read_csv(raw / "movies.dat", sep="::", header=None, engine="python", encoding="ISO-8859-1",
                         names=["movie_id", "title", "genres"])

    frame = pd.merge(pd.merge(ratings, movies), users)
    total = len(frame)
    frame = frame[frame["rating"] != 3].copy()
    frame["label"] = (frame["rating"] >= 4).astype(np.int32)
    frame["cate_id"] = frame["genres"].str.split("|").str[0]
    print(f"ratings {total} -> {len(frame)} after removing 3-star", flush=True)

    permutation = np.random.default_rng(args.seed).permutation(len(frame))
    test_size = len(frame) // 10
    valid_size = len(frame) // 10
    order = {
        "valid": permutation[:valid_size],
        "test": permutation[valid_size:valid_size + test_size],
        "train": permutation[valid_size + test_size:],
    }

    # Factorize each field on the train split only, then map every split.
    encodings: list[tuple[np.ndarray, np.ndarray]] = []
    train_frame = frame.iloc[order["train"]]
    for column in CATEGORICAL:
        tokens, counts = np.unique(train_frame[column].to_numpy(), return_counts=True)
        rank = np.lexsort((tokens.astype(str), -counts))
        ids = np.empty(len(tokens), dtype=np.int64)
        ids[rank] = np.arange(1, len(tokens) + 1)
        encodings.append((tokens, ids))
    vocab_sizes = [len(tokens) + 1 for tokens, _ in encodings]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for split, index in order.items():
        part = frame.iloc[index]
        sparse = np.empty((len(part), len(CATEGORICAL)), dtype=np.int32)
        oov = {}
        for position, (column, (tokens, ids)) in enumerate(zip(CATEGORICAL, encodings)):
            values = part[column].to_numpy()
            found = np.searchsorted(tokens.astype(str), values.astype(str))
            found = np.clip(found, 0, len(tokens) - 1)
            hit = tokens.astype(str)[found] == values.astype(str)
            sparse[:, position] = np.where(hit, ids[found], 0)
            oov[column] = float((sparse[:, position] == 0).mean())
        dense = np.zeros((len(part), 0), dtype=np.float32)
        np.savez(output / f"{split}.npz", label=part["label"].to_numpy(np.int32), dense=dense, sparse=sparse)
        rows[split] = {"rows": len(part), "positive_ratio": float(part["label"].mean()), "oov_rate": oov}
        print(split, dense.shape, sparse.shape, "positive", round(rows[split]["positive_ratio"], 4), flush=True)

    meta = {
        "name": "movielens_1m_ctr",
        "num_numeric": 0,
        "vocab_sizes": vocab_sizes,
        "fields": CATEGORICAL,
        "total_vocab": int(sum(vocab_sizes)),
        "rows": {split: value["rows"] for split, value in rows.items()},
        "positive_ratio": {split: value["positive_ratio"] for split, value in rows.items()},
        "oov_rate": {split: value["oov_rate"] for split, value in rows.items()},
        "split": f"random 80/10/10, seed {args.seed}; remove 3-star, 1/2->0, 4/5->1 (paper 7.1.1)",
        "deviations": "paper does not publish the split seed or the exact feature list",
        "source": str(raw.resolve()),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("vocab_sizes", "oov_rate")}, indent=2))


if __name__ == "__main__":
    main()
