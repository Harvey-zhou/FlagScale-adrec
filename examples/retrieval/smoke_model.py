"""One-batch smoke test for a native FlagScale retrieval adapter."""

from __future__ import annotations

import argparse

import os
import sys

# Torch-FL must claim the PrivateUse1 slot before torch builds a flagos tensor,
# and torch_mlu claims the same slot, so only the requested one is imported.
_DEVICE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--device=")), None)
if _DEVICE is None and "--device" in sys.argv:
    _i = sys.argv.index("--device")
    _DEVICE = sys.argv[_i + 1] if _i + 1 < len(sys.argv) else None
_DEVICE = (_DEVICE or os.environ.get("FLAGSCALE_DEVICE", "flagos")).strip().lower()
if _DEVICE.startswith("flagos"):
    import torch_fl  # noqa: F401  # must precede torch
import torch
from omegaconf import OmegaConf

from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch


_MODELS = {
    "qwen3_embedding": {
        "task": "embedding",
        "path": "/flagos-retrieval-models/Qwen3-Embedding-0.6B",
        "data": "/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
        "max_length": 128,
        "num_negatives": 1,
    },
    "clip_vit_large": {
        "task": "clip",
        "path": "/flagos-retrieval-models/clip-vit-large-patch14/clip-vit-large-patch14",
        "data": "/flagos-retrieval-datasets/wds_mscoco_captions2017/wds_mscoco_captions2017/train",
        "max_length": 77,
        "num_negatives": 1,
    },
    "qwen3_reranker": {
        "task": "reranker",
        "path": "/flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B",
        "data": "/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
        "max_length": 128,
        "num_negatives": 1,
        "freeze_backbone": True,
        "use_score_head": True,
    },
    "qwen3_vl_embedding": {
        "task": "vl_embedding",
        "path": "/flagos-retrieval-models/Qwen3-VL-Embedding-8B/Qwen3-VL-Embedding-8B",
        "data": "/flagos-retrieval-datasets/wds_mscoco_captions2017/wds_mscoco_captions2017/train",
        # Image placeholders expand to a few hundred tokens; do not truncate
        # them in the processor smoke test.
        "max_length": 2048,
        "num_negatives": 1,
        "freeze_backbone": True,
        "projection_dim": 1024,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(_MODELS))
    parser.add_argument("--device", default=_DEVICE, help="flagos | mlu | npu | cuda (default flagos)")
    args = parser.parse_args()
    spec = _MODELS[args.model]
    requested = args.device.partition(":")[0]
    if requested in {"cuda", "flagos"}:
        device = torch.device(f"{requested}:0")
    else:
        from flagscale.platforms.platform_register import PLATFORMS

        if requested not in PLATFORMS:
            raise RuntimeError(f"--device {args.device!r} unavailable; registered: {sorted(PLATFORMS)}")
        PLATFORMS[requested].set_device(0)
        device = PLATFORMS[requested].device(0)
    model_cfg = OmegaConf.create(
        {
            "model_type": args.model,
            "model_path": spec["path"],
            "max_length": spec["max_length"],
            "torch_dtype": "bf16",
            "freeze_backbone": bool(spec.get("freeze_backbone", False)),
            "use_score_head": bool(spec.get("use_score_head", False)),
            "projection_dim": int(spec.get("projection_dim", 0)),
        }
    )
    print(f"loading {args.model}", flush=True)
    model, processor = build_retrieval_model(args.model, model_cfg, device)
    print(f"loaded parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    data_cfg = OmegaConf.create(
        {
            "path": spec["data"],
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "num_negatives": spec["num_negatives"],
            "split_modulo": 20,
            "max_samples": 2,
            "eval_max_samples": 2,
        }
    )
    loader, _, _ = build_split_loaders(spec["task"], data_cfg, 2, pin_memory=True)
    batch = next(iter(loader))
    print(f"batch keys={sorted(batch)}", flush=True)
    loss = loss_for_batch(model, processor, spec["task"], batch, device, 0.05)
    print(f"loss={float(loss.detach().cpu()):.6f}", flush=True)
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.step()
    print("backward and AdamW step passed", flush=True)


if __name__ == "__main__":
    main()
