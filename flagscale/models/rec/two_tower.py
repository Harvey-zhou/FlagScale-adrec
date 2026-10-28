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

"""Two-tower retrieval model, ported from TorchRec ``examples/retrieval``.

TorchRec's example (``modules/two_tower.py`` and ``two_tower_train.py``) embeds a query id
and a candidate id with one embedding-bag table each, runs every tower through an MLP and
scores the pair with an inner product under binary cross-entropy.  This module re-implements
that recipe with plain ``torch.nn`` modules, so it needs neither TorchRec's sharding runtime,
nor FBGEMM kernels, nor FAISS:

* two tables of ``num_embeddings`` x ``embedding_dim``, initialised U(-sqrt(1/n), sqrt(1/n))
  as ``EmbeddingBagConfig`` does by default.  With one id per feature the bag sum is a plain
  lookup, and an empty bag is the zero vector;
* a ReLU MLP of ``layer_sizes`` per tower.  TorchRec's ``MLP`` applies the activation after
  every layer, the last one included;
* logits are the inner product of the two tower outputs, trained with BCE-with-logits;
* optimizer: row-wise Adagrad on the tables (the example applies it in the backward pass)
  and Adam on the MLPs.

TorchRec's example trains on random data, so this workload has no accuracy target.  It
checks that lookup, MLP, loss, backward and the optimizers run and agree across devices.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from .base import RecModel
from .registry import register_rec_model


class RowWiseAdagrad(torch.optim.Optimizer):
    """Port of ``torchrec.optim.RowWiseAdagrad``: one Adagrad accumulator per embedding row.

    The accumulator of a row adds the mean of the squared gradient over that row, so the
    adaptive step size is shared by the whole row.  Dense gradients only.
    """

    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0.0, weight_decay: float = 0.0,
                 eps: float = 1e-10) -> None:
        super().__init__(params, dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps))
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param]["step"] = 0
                self.state[param]["sum"] = torch.zeros(param.shape[0], 1, dtype=param.dtype, device=param.device)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("RowWiseAdagrad does not support sparse gradients")
                state = self.state[param]
                state["step"] += 1
                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])
                state["sum"] += grad.pow(2).mean(dim=1, keepdim=True)
                std = state["sum"].sqrt().add_(group["eps"])
                rate = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])
                param.addcdiv_(grad, std, value=-rate)
        return None


class CompositeOptimizer:
    """Runs several optimizers as one, for models that use a different recipe per parameter group."""

    def __init__(self, *optimizers: torch.optim.Optimizer) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def state_dict(self) -> list[dict[str, Any]]:
        return [optimizer.state_dict() for optimizer in self.optimizers]


def _mlp(in_dim: int, layer_sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in layer_sizes:
        layers += [nn.Linear(in_dim, width), nn.ReLU()]
        in_dim = width
    return nn.Sequential(*layers)


@register_rec_model("two_tower")
class TwoTower(RecModel):
    """Query / candidate towers; ``forward`` returns the inner-product logits of each pair."""

    task = "synthetic_two_tower"

    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__(model_cfg, meta)
        dim = int(model_cfg.get("embedding_dim", 64))
        rows = int(meta["num_embeddings"])
        sizes = [int(size) for size in model_cfg.get("layer_sizes", [128, 64])]
        self.query_embedding = nn.Embedding(rows, dim)
        self.candidate_embedding = nn.Embedding(rows, dim)
        self.query_proj = _mlp(dim, sizes)
        self.candidate_proj = _mlp(dim, sizes)
        bound = math.sqrt(1.0 / rows)
        for table in (self.query_embedding, self.candidate_embedding):
            nn.init.uniform_(table.weight, -bound, bound)

    @staticmethod
    def _pooled(table: nn.Embedding, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # One id per feature: the bag sum is the lookup, and a sample without an id gets zeros.
        return table(ids) * lengths.unsqueeze(-1).to(table.weight.dtype)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        query = self.query_proj(self._pooled(self.query_embedding, batch["user_id"], batch["user_len"]))
        candidate = self.candidate_proj(
            self._pooled(self.candidate_embedding, batch["movie_id"], batch["movie_len"]))
        return (query * candidate).sum(dim=1)

    def configure_optimizer(self, optim_cfg: DictConfig) -> CompositeOptimizer:
        tables = [self.query_embedding.weight, self.candidate_embedding.weight]
        table_ids = {id(parameter) for parameter in tables}
        dense = [parameter for parameter in self.parameters() if id(parameter) not in table_ids]
        lr = float(optim_cfg.lr)
        return CompositeOptimizer(
            RowWiseAdagrad(tables, lr=float(optim_cfg.get("embedding_lr", lr))),
            torch.optim.Adam(dense, lr=lr),
        )
