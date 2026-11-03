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

"""Cambricon MLU adapter: vendor ``torch_mlu`` + CNCL, without Torch-FL.

``torch_mlu`` renames PyTorch's PrivateUse1 slot to ``mlu`` and registers the
``cncl`` process-group backend, so this adapter drives it directly, the way
``platform_npu.py`` drives ``torch_npu``.

``torch_mlu.utils.gpu_migration`` is deliberately not used.  It rewrites the
``torch.cuda`` namespace onto the MLU, so a run would report ``cuda`` while
executing on an MLU, which is the kind of mislabelled result this project
rules out.
"""

import torch

from .platform_base import PlatformBase


def _privateuse1_taken_by_other_backend() -> bool:
    """True when another backend (e.g. Torch-FL's ``flagos``) owns PrivateUse1.

    Importing ``torch_mlu`` then fails half-way, after it has already patched
    torch, so the import is not attempted at all.
    """
    try:
        name = torch._C._get_privateuse1_backend_name()
    except AttributeError:
        return False
    return name not in ("privateuseone", "mlu")


class PlatformMLU(PlatformBase):
    def name(self) -> str:
        return "mlu"

    def is_available(self) -> bool:
        if _privateuse1_taken_by_other_backend():
            return False
        try:
            import torch_mlu  # noqa: F401

            return torch.mlu.is_available() and torch.mlu.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        import torch_mlu  # noqa: F401

        torch.mlu.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("mlu", device_index)

    def device_count(self) -> int:
        import torch_mlu  # noqa: F401

        return torch.mlu.device_count()

    def dist_backend(self) -> str:
        return "cncl"

    def manual_seed_all(self, seed):
        import torch_mlu  # noqa: F401

        torch.mlu.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "mlu"

    # supports_pin_memory() keeps the base default (False) until page-locked host
    # memory is confirmed on the cluster: run-adrec-cambricon.sh's preflight
    # reports it, and FLAGSCALE_PIN_MEMORY=true turns it on.
