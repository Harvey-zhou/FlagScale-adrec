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

"""Convert a LightGCN dataset (Gowalla, Yelp2018 or Amazon-Book) to ``.npz``.

Input is the authors' split, e.g. ``data/<dataset>/{train,test}.txt`` of
gusye1234/LightGCN-PyTorch (the same files as kuandeng/LightGCN ``Data/``).  Each line is
``user item item ...``.  No filtering or re-splitting is applied; the script only checks
the published statistics, so a wrong or truncated download fails loudly.

The user, item and total-interaction counts are those of the LightGCN paper (Table 2).  The
train / test counts were measured on the distributed files.

    python examples/rec/prepare_lightgcn.py --dataset yelp2018 --raw <LightGCN-PyTorch>/data/yelp2018
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# name -> (output directory, expected statistics)
DATASETS = {
    "gowalla": ("gowalla_lightgcn", {"n_users": 29858, "n_items": 40981, "train": 810128, "test": 217242}),
    "yelp2018": ("yelp2018_lightgcn", {"n_users": 31668, "n_items": 38048, "train": 1237259, "test": 324147}),
    "amazon-book": ("amazon_book_lightgcn", {"n_users": 52643, "n_items": 91599, "train": 2380730,
                                             "test": 603378}),
}


def read_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    users, items = [], []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            user = int(tokens[0])
            for token in tokens[1:]:
                users.append(user)
                items.append(int(token))
    return np.asarray(users, dtype=np.int32), np.asarray(items, dtype=np.int32)


def main() -> None:
    root = os.environ.get("FLAGOS_REC_DATA", "/flagos-rec/datasets")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--raw", default=None, help="directory with train.txt and test.txt "
                                                    "(default: <root>/raw/lightgcn/<dataset>)")
    parser.add_argument("--output", default=None, help="default: <root>/<dataset>_lightgcn; "
                                                       "<root> is $FLAGOS_REC_DATA")
    args = parser.parse_args()
    directory, expected = DATASETS[args.dataset]
    raw = Path(args.raw or f"{root}/raw/lightgcn/{args.dataset}")
    train_users, train_items = read_pairs(raw / "train.txt")
    test_users, test_items = read_pairs(raw / "test.txt")
    n_users = int(max(train_users.max(), test_users.max()) + 1)
    n_items = int(max(train_items.max(), test_items.max()) + 1)
    stats = {"n_users": n_users, "n_items": n_items, "train": len(train_users), "test": len(test_users)}
    mismatch = {k: (stats[k], v) for k, v in expected.items() if stats[k] != v}
    if mismatch:
        raise ValueError(f"{args.dataset} statistics differ from the published split: {mismatch}")
    output = Path(args.output or f"{root}/{directory}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "train.npz", user=train_users, item=train_items)
    np.savez(output / "test.npz", user=test_users, item=test_items)
    meta = {"name": f"{args.dataset}-lightgcn", "task": "graph", **stats, "source": str(raw.resolve())}
    with (output / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
