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

"""Common contract for the native advertising (CTR) models.

The models reproduce FuxiCTR implementations benchmarked in BARS.  Two
FuxiCTR conventions are part of the contract:

* every parameter whose name contains ``embedding_layer`` receives the
  embedding L2 regularizer ``lambda / 2 * ||W||^2`` (BaseModel.add_regularization);
* ``nn.Linear`` weights are Xavier-normal with zero bias and embedding tables
  are N(0, 1e-4) (BaseModel.reset_parameters); padding rows stay zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class AdsModel(nn.Module):
    """Binary CTR model; ``forward`` returns logits."""

    task = "ctr"

    def __init__(self, model_cfg: DictConfig, meta: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.meta = meta

    @classmethod
    def from_config(cls, model_cfg: DictConfig, meta: dict[str, Any], device: torch.device) -> "AdsModel":
        return cls(model_cfg, meta).to(device)

    def reset_parameters(self) -> None:
        """Initialise weights the FuxiCTR way, or the authors' TF way on request.

        ``model.embedding_init`` selects the embedding-table initialiser:

        * ``normal`` (default): N(0, 1e-4).  This is FuxiCTR's convention and
          what every BARS config in this repository was trained with;
        * ``glorot_uniform``: Xavier/Glorot uniform, the TensorFlow default that
          the DIN authors' ``tf.get_variable`` tables get.  For a 63,003 x 64
          table the bound is ~0.0098, i.e. about 100x the standard deviation of
          ``normal``, so this is a materially different starting point for SGD.

        Dense layers keep FuxiCTR's Xavier-normal initialisation either way.
        """

        embedding_init = str(self.model_cfg.get("embedding_init", "normal")).lower()
        if embedding_init not in {"normal", "glorot_uniform", "xavier_uniform"}:
            raise ValueError(f"Unsupported embedding_init {embedding_init!r}")
        for module in self.modules():
            if type(module) is nn.Linear:
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
            elif type(module) is nn.Embedding:
                with torch.no_grad():
                    if module.padding_idx is not None:
                        rows = torch.tensor(
                            [i for i in range(module.num_embeddings) if i != module.padding_idx],
                            dtype=torch.long, device=module.weight.device)
                    else:
                        rows = torch.arange(module.num_embeddings, device=module.weight.device)
                    if embedding_init == "normal":
                        module.weight[rows] = torch.randn(
                            len(rows), module.embedding_dim, device=module.weight.device) * 1e-4
                    else:
                        fresh = torch.empty(len(rows), module.embedding_dim, device=module.weight.device)
                        nn.init.xavier_uniform_(fresh)
                        module.weight[rows] = fresh
                    if module.padding_idx is not None:
                        module.weight[module.padding_idx].zero_()

    def embedding_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if "embedding_layer" in name and p.requires_grad]

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch))

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.state_dict().items()}, output_dir / "model.pt")
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump({"model": OmegaConf.to_container(self.model_cfg, resolve=True), "meta": self.meta},
                      file, indent=2)
