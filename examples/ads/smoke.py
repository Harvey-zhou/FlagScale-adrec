"""Few-step smoke test for an ads model on cpu, cuda, Torch-FL or a Cambricon MLU.

    python examples/ads/smoke.py --model din --device flagos --steps 3
    python examples/ads/smoke.py --config din_paper --device mlu
"""

from __future__ import annotations

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=["din", "deepfm", "dcn_mix"], default=None)
parser.add_argument("--config", default=None, help="name of conf/train/<name>.yaml (default: the model name)")
parser.add_argument("--device", default="cpu", help="cpu, cuda, flagos or mlu")
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()
if not (args.model or args.config):
    parser.error("give --model or --config")
os.environ["FLAGSCALE_DEVICE"] = args.device
if args.device.startswith("flagos"):
    import torch_fl  # noqa: F401  # must be imported before torch
elif args.device.startswith("mlu"):
    import torch_mlu  # noqa: F401  # registers the mlu device

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from flagscale.models.ads import build_ads_model  # noqa: E402
from flagscale.train import train_ads  # noqa: E402
from flagscale.train.datasets.ads import build_ads_loaders  # noqa: E402
from flagscale.train.losses.ads import ctr_loss  # noqa: E402

CONF = os.path.join(os.path.dirname(__file__), "conf", "train")


def main() -> None:
    name = args.config or args.model
    cfg = OmegaConf.load(os.path.join(CONF, f"{name}.yaml"))
    cfg.data.max_samples = int(cfg.batch_size) * args.steps
    cfg.data.eval_max_samples = 20000
    device = torch.device(f"{args.device}:0" if args.device in {"cuda", "flagos", "mlu"} else args.device)
    train_loader, valid_loader, _, meta = build_ads_loaders(cfg.data, int(cfg.batch_size), 8192, False, 0)
    model = build_ads_model(str(cfg.model.model_type), cfg.model, meta, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr))
    params = model.embedding_parameters()
    print(f"{name} parameters={sum(p.numel() for p in model.parameters())} device={device}", flush=True)
    model.train()
    for step, batch in enumerate(train_loader, start=1):
        loss = ctr_loss(model, batch, device, params, float(cfg.optimizer.embedding_regularizer))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        print(f"step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
    result = train_ads.evaluate(model, valid_loader, device, "fp32", None, cfg.data.get("group_key"))
    print("eval", {k: round(v, 4) for k, v in result.items()})
    print("SMOKE OK")


if __name__ == "__main__":
    main()
