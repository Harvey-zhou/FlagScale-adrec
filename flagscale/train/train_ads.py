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

#!/usr/bin/env python3
"""Training entrypoint for the native advertising (CTR) models (examples/ads).

Launched by FlagScale's ``native`` backend: Hydra composes the YAML, torchrun
starts this module with ``--config-file``, and the device is selected through
the platform interface, so native CUDA and Torch-FL differ only in a
launch-time override.

The loop supports two protocols, selected by ``optimizer.name`` /
``optimizer.scheduler.type`` in the config:

* ``scheduler.type: plateau`` (default) reproduces FuxiCTR's BaseModel fit used
  by the BARS benchmarks: Adam; loss = BCE + ``embedding_regularizer / 2 *
  ||W_emb||^2``; gradient norm clipped to 10; after every epoch the validation
  split is scored, a non-improvement (< 1e-6) multiplies the learning rate by
  0.1 and increases a patience counter, and training stops when the counter
  reaches ``early_stop_patience``.
* ``scheduler.type: step`` (or ``none`` / ``exponential``) runs a fixed
  schedule instead: the learning rate is ``lr * gamma ** (step //
  step_size_steps)`` with optional linear ``warmup_steps``; the monitored
  metric still selects the best checkpoint, but it never changes the learning
  rate. Set ``early_stop_patience: 0`` to run every configured epoch.

In both cases the best checkpoint by the monitored split is restored and
scored on the test split.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

# torch-fl must claim PyTorch's PrivateUse1 slot before the first flagos tensor
# is created; see train_retrieval.py for the same opt-in import.
_REQUESTED_DEVICE = os.environ.get("FLAGSCALE_DEVICE", os.environ.get("FS_PLATFORM", "auto")).strip().lower()
if _REQUESTED_DEVICE in {"flagos", "privateuseone"}:
    try:
        import torch_fl  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - vendor envs
        raise RuntimeError("FLAGSCALE_DEVICE=flagos requires a matching torch-fl installation.") from exc

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from flagscale.models.ads import build_ads_model
from flagscale.train.datasets.ads import build_ads_loaders
from flagscale.train.losses.ads import ctr_loss, move_batch
from flagscale.train.metrics.ads import ctr_metrics

try:
    from flagscale.platforms import get_platform
except ImportError:  # pragma: no cover - direct standalone invocation
    get_platform = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, help="Hydra-generated or standalone YAML")
    return parser.parse_args()


def rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if rank_zero():
        print(message, flush=True)


def init_distributed(hardware: DictConfig | None = None) -> tuple[int, int, torch.device, Any]:
    """Select the device through the platform interface and join torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    hardware = hardware or {}
    requested = str(
        hardware.get("device_type", os.environ.get("FS_PLATFORM", os.environ.get("FLAGSCALE_DEVICE", "auto")))
    ).lower()
    platform = None
    if get_platform is not None:
        try:
            platform = get_platform()
        except (RuntimeError, ValueError):
            platform = None
    if requested in {"privateuseone", "flagos"}:
        if platform is None or platform.name() != "flagos":
            raise RuntimeError("device_type=flagos was requested, but a Torch-FL platform is unavailable")
    elif requested == "cpu":
        platform = None
    elif requested not in {"auto", ""} and (platform is None or platform.name() != requested):
        raise RuntimeError(
            f"Requested platform {requested!r} is unavailable; the active runtime exposes "
            f"{platform.name() if platform is not None else 'cpu'!r}"
        )
    if platform is not None and (requested not in {"auto", ""} or platform.is_available()):
        platform.set_device(local_rank)
        device = platform.device(local_rank)
    else:
        platform = None
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = str(hardware.get("distributed_backend", "auto")).lower()
        if backend == "auto":
            backend = platform.dist_backend() if platform is not None else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, device, platform


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, rank: int, platform: Any = None) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if platform is not None:
        platform.manual_seed_all(seed)


def autocast_context(device: torch.device, dtype_name: str, platform: Any = None):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    autocast_device = platform.amp_device_type() if platform is not None else device.type
    try:
        return torch.autocast(device_type=autocast_device, dtype=dtype)
    except (RuntimeError, TypeError):
        return contextlib.nullcontext()


def all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    # A single process has nothing to reduce, so no tensor has to touch the device
    # (vendor runtimes do not all implement float64).  The multi-rank path uses
    # float32: the values are only logged statistics (mean loss, throughput).
    if not (dist.is_available() and dist.is_initialized()):
        return [float(v) for v in values]
    tensor = torch.tensor(values, dtype=torch.float32).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def resolve_pin_memory(hardware: Any, platform: Any) -> bool:
    """``train.hardware.pin_memory``: true / false, or null / "auto" for the platform default.

    Some vendor runtimes report the CUDA platform without supporting page-locked host
    memory, so the platform default can be overridden from the launch command.
    """

    value = hardware.get("pin_memory", None) if hardware else None
    if isinstance(value, str):
        value = value.strip().lower()
        value = None if value in {"", "null", "none", "auto"} else value in {"1", "true", "yes", "on"}
    if value is None:
        return bool(platform and platform.supports_pin_memory())
    return bool(value)


class ExperimentLogger:
    """Optional Weights & Biases run on rank zero.

    ``WANDB_MODE=offline`` writes the run under ``<output_dir>/wandb`` so it can
    be uploaded later with ``wandb sync``; ``WANDB_DISABLED=true`` or a missing
    wandb package turns logging off without affecting training.
    """

    def __init__(self, train_cfg: DictConfig, output_dir: Path, extra: dict[str, Any]) -> None:
        self.run = None
        if not rank_zero() or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            return
        try:
            import wandb
        except ImportError:
            log("wandb is not installed; continuing without experiment logging")
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        config = OmegaConf.to_container(train_cfg, resolve=True)
        config.update(extra)
        self.run = wandb.init(
            project=str(train_cfg.get("wandb_project", "flagos-ads")),
            name=str(train_cfg.get("wandb_name", f"{train_cfg.name}-{extra.get('device_type')}")),
            group=str(train_cfg.name),
            tags=[str(train_cfg.name), str(extra.get("device_type"))],
            dir=str(output_dir),
            config=config,
        )
        log(f"W&B run {self.run.id} mode={os.environ.get('WANDB_MODE', 'online')} dir={output_dir / 'wandb'}")

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            for key, value in values.items():
                self.run.summary[key] = value

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def train_epoch(model: nn.Module, loader, epoch: int, device: torch.device, optimizer: torch.optim.Optimizer,
                embedding_parameters: list[nn.Parameter], embedding_regularizer: float, max_grad_norm: float,
                dtype_name: str, platform: Any, log_every: int,
                schedule: "LRSchedule | None" = None, start_step: int = 0) -> dict[str, float]:
    model.train()
    sampler = getattr(loader.sampler, "sampler", loader.sampler)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    total: torch.Tensor | None = None
    steps = 0
    samples = 0
    started = time.perf_counter()
    for batch in loader:
        samples += int(batch["label"].shape[0])
        if schedule is not None:
            learning_rate = schedule.lr_at(start_step + steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        with autocast_context(device, dtype_name, platform):
            loss = ctr_loss(model, batch, device, embedding_parameters, embedding_regularizer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        detached = loss.detach().float()
        total = detached if total is None else total + detached
        steps += 1
        if log_every > 0 and steps % log_every == 0:
            log(f"epoch={epoch} step={steps} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"running_loss={float(total.item()) / steps:.6f}")
    if steps == 0:
        raise RuntimeError("Training dataset is empty for this rank")
    local_loss = float(total.item())
    elapsed = time.perf_counter() - started
    loss_sum, step_sum, sample_sum = all_reduce_sum([local_loss, float(steps), float(samples)], device)
    return {"loss": loss_sum / max(step_sum, 1.0), "steps": steps, "elapsed_s": elapsed,
            "samples_per_s": sample_sum / max(elapsed, 1e-9), "end_step": start_step + steps,
            "end_lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, dtype_name: str, platform: Any,
             group_key: str | None) -> dict[str, float]:
    model.eval()
    scores, labels, groups = [], [], []
    started = time.perf_counter()
    for batch in loader:
        with autocast_context(device, dtype_name, platform):
            scores.append(model.predict(move_batch(batch, device)).float().cpu().numpy())
        labels.append(batch["label"].numpy())
        if group_key:
            groups.append(batch[group_key].numpy())
    result = ctr_metrics(np.concatenate(labels), np.concatenate(scores),
                         np.concatenate(groups) if group_key else None)
    result["elapsed_s"] = time.perf_counter() - started
    return result


def fmt(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.6f}" for k, v in metrics.items() if k != "elapsed_s")


def make_optimizer(model: nn.Module, optim_cfg: DictConfig) -> torch.optim.Optimizer:
    """Adam (FuxiCTR default) or plain SGD with optional momentum/weight decay."""

    name = str(optim_cfg.get("name", "adam")).lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=float(optim_cfg.get("momentum", 0.0)),
            weight_decay=weight_decay, nesterov=bool(optim_cfg.get("nesterov", False)))
    raise ValueError(f"Unsupported optimizer {name!r}; use 'adam' or 'sgd'")


class LRSchedule:
    """Step / exponential / constant schedule, optionally with linear warmup.

    ``step`` multiplies the learning rate by ``gamma`` every
    ``step_size_steps`` optimizer steps, which is how the DIN authors' released
    code drops 1.0 -> 0.1 at 336,000 steps.  ``exponential`` multiplies it by
    ``gamma`` once per epoch.  ``plateau`` is not scheduled here: the epoch loop
    keeps FuxiCTR's non-improvement decay for that mode.
    """

    def __init__(self, scheduler_cfg: DictConfig | None, base_lr: float) -> None:
        scheduler_cfg = scheduler_cfg or {}
        self.type = str(scheduler_cfg.get("type", "plateau")).lower()
        if self.type not in {"plateau", "none", "step", "exponential"}:
            raise ValueError(f"Unsupported scheduler {self.type!r}")
        self.gamma = float(scheduler_cfg.get("gamma", 0.1))
        self.step_size = int(scheduler_cfg.get("step_size_steps", 0) or 0)
        self.warmup = int(scheduler_cfg.get("warmup_steps", 0) or 0)
        self.min_lr = float(scheduler_cfg.get("min_lr", 0.0) or 0.0)
        self.base_lr = float(base_lr)
        if self.type == "step" and self.step_size <= 0:
            raise ValueError("scheduler.type=step needs a positive step_size_steps")

    def lr_at(self, step: int) -> float:
        """Learning rate for the next optimizer step (``step`` = steps done so far)."""

        multiplier = 1.0
        if self.warmup > 0 and step < self.warmup:
            multiplier = (step + 1) / self.warmup
        elif self.type == "step":
            multiplier = self.gamma ** ((step - self.warmup) // self.step_size)
        lr = self.base_lr * multiplier
        return max(lr, self.min_lr) if self.min_lr > 0 else lr

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "gamma": self.gamma, "step_size_steps": self.step_size,
                "warmup_steps": self.warmup, "min_lr": self.min_lr, "base_lr": self.base_lr}



def train(config: DictConfig) -> None:
    train_cfg = config.train
    rank, world_size, device, platform = init_distributed(train_cfg.get("hardware", {}))
    seed = int(train_cfg.get("seed", 2021))
    seed_everything(seed, rank, platform)
    model_cfg = train_cfg.model
    dtype_name = str(model_cfg.get("torch_dtype", "fp32"))
    train_loader, valid_loader, test_loader, meta = build_ads_loaders(
        train_cfg.data, int(train_cfg.batch_size), int(train_cfg.get("eval_batch_size", train_cfg.batch_size)),
        pin_memory=resolve_pin_memory(train_cfg.get("hardware", {}), platform), seed=seed)
    model = build_ads_model(str(model_cfg.model_type), model_cfg, meta, device)
    base_model = model
    parameters = sum(p.numel() for p in model.parameters())
    device_type = platform.name() if platform is not None else "cpu"
    valid_split = str(train_cfg.data.get("valid_split", "valid"))
    log(f"model={model_cfg.model_type} parameters={parameters} device={device} world_size={world_size}")
    log(f"dataset={meta.get('name')} train={len(train_loader.dataset)} valid={len(valid_loader.dataset)} "
        f"test={len(test_loader.dataset)} valid_split={valid_split}")
    embedding_parameters = base_model.embedding_parameters()
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device_type == "cuda" else None)

    optim_cfg = train_cfg.optimizer
    optimizer = make_optimizer(model, optim_cfg)
    scheduler_cfg = train_cfg.get("scheduler")
    if scheduler_cfg is None:
        scheduler_cfg = optim_cfg.get("scheduler")
    schedule = LRSchedule(scheduler_cfg, float(optim_cfg.lr))
    fixed_schedule = schedule.type != "plateau"
    if fixed_schedule and schedule.warmup == 0:
        # No warmup: the schedule owns the learning rate from the first step, so
        # an optimizer constructed with a different lr cannot leak in.
        for group in optimizer.param_groups:
            group["lr"] = schedule.lr_at(0)
    embedding_regularizer = float(optim_cfg.get("embedding_regularizer", 0.0))
    max_grad_norm = float(optim_cfg.get("max_gradient_norm", 10.0))
    epochs = int(train_cfg.get("epochs", 100))
    patience = int(train_cfg.get("early_stop_patience", 2))
    monitor = str(train_cfg.get("monitor", "AUC"))
    group_key = train_cfg.data.get("group_key")
    log_every = int(train_cfg.get("log_every", 0))
    output_dir = Path(str(train_cfg.output_dir))
    logger = ExperimentLogger(train_cfg, output_dir, {"device_type": device_type, "parameters": parameters,
                                                      "torch_version": torch.__version__, "dataset_meta": meta})
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"epoch": 0, "value": float("-inf"), "state": None, "valid": None}
    stopping_steps = 0
    global_step = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start_lr = float(optimizer.param_groups[0]["lr"])
        stats = train_epoch(model, train_loader, epoch, device, optimizer, embedding_parameters,
                            embedding_regularizer, max_grad_norm, dtype_name, platform, log_every,
                            schedule if fixed_schedule else None, global_step)
        global_step = int(stats["end_step"])
        record: dict[str, Any] = {"epoch": epoch, "train": stats, "lr": epoch_start_lr,
                                  "end_lr": stats["end_lr"], "global_step": global_step}
        # [stop flag, reduce-lr flag] decided on rank zero and shared.  The new learning
        # rate is derived on every rank from the flag, so no float64 value has to be
        # broadcast through the device.
        decision = torch.zeros(2, dtype=torch.float32)
        if rank_zero():
            valid = evaluate(base_model, valid_loader, device, dtype_name, platform, group_key)
            record["valid"] = valid
            value = float(valid[monitor])
            if value < best["value"] + 1e-6:
                stopping_steps += 1
                if not fixed_schedule:
                    decision[1] = 1.0
                    next_lr = max(optimizer.param_groups[0]["lr"] * 0.1, 1e-6)
                    log(f"monitor {monitor}={value:.6f} did not improve; lr -> {next_lr:.6g}")
            else:
                stopping_steps = 0
                best = {"epoch": epoch, "value": value, "valid": valid,
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in base_model.state_dict().items()})}
                # Keep the best-so-far checkpoint on disk: a long run that is
                # interrupted still leaves a usable model behind.
                base_model.save(output_dir)
                log(f"new best {monitor}={value:.6f} at epoch {epoch}; checkpoint saved")
            if patience > 0 and stopping_steps >= patience:
                decision[0] = 1.0
            logger.log({"train/loss": stats["loss"], "train/samples_per_s": stats["samples_per_s"],
                        "train/epoch_time_s": stats["elapsed_s"], "train/lr": epoch_start_lr,
                        **{f"valid/{k}": v for k, v in valid.items()}}, step=epoch)
            log(f"epoch={epoch} loss={stats['loss']:.6f} samples_per_s={stats['samples_per_s']:.0f} "
                f"lr={epoch_start_lr:.6g} step={global_step} valid: {fmt(valid)}")
        if dist.is_available() and dist.is_initialized():
            shared = decision.to(device)
            dist.broadcast(shared, src=0)
            decision = shared.cpu()
        if decision[1].item() > 0:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * 0.1, 1e-6)
        history.append(record)
        if rank_zero():
            # Incremental history: `metrics.json` is only written once the run
            # finishes, and these runs are hours long.  Write-then-rename so a
            # kill in the middle of the dump cannot leave a truncated file.
            output_dir.mkdir(parents=True, exist_ok=True)
            temporary = output_dir / "history.json.tmp"
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(history, file, indent=2)
            temporary.replace(output_dir / "history.json")
        if decision[0].item() > 0:
            log(f"early stop at epoch {epoch}")
            break
        if schedule.type == "exponential":
            schedule.base_lr *= schedule.gamma

    if rank_zero():
        base_model.load_state_dict(best["state"])
        test = evaluate(base_model, test_loader, device, dtype_name, platform, group_key)
        log(f"best_epoch={best['epoch']} test: {fmt(test)} elapsed={time.time() - started:.1f}s")
        base_model.save(output_dir)
        rates = [r["train"]["samples_per_s"] for r in history]
        summary = {
            "model": str(model_cfg.model_type), "dataset": meta.get("name"), "device": str(device),
            "device_type": device_type, "world_size": world_size, "torch_version": torch.__version__,
            "parameters": parameters, "monitor": monitor, "valid_split": valid_split,
            "optimizer": {"name": str(optim_cfg.get("name", "adam")), "lr": float(optim_cfg.lr),
                          "momentum": float(optim_cfg.get("momentum", 0.0)),
                          "weight_decay": float(optim_cfg.get("weight_decay", 0.0)),
                          "embedding_regularizer": embedding_regularizer,
                          "max_gradient_norm": max_grad_norm},
            "scheduler": schedule.to_dict(),
            "best_epoch": best["epoch"], "valid": best["valid"], "test": test,
            "train_samples_per_s_median": float(np.median(rates)), "elapsed_s": time.time() - started,
            "history": history,
        }
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        logger.summary({f"test/{k}": v for k, v in test.items()})
        logger.summary({"best_epoch": best["epoch"], "parameters": parameters, "elapsed_s": summary["elapsed_s"],
                        "train_samples_per_s_median": summary["train_samples_per_s_median"]})
        log(f"saved checkpoint and metrics to {output_dir}")
    logger.finish()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_file)
    if not config.get("train"):
        raise ValueError("Configuration must contain a train section")
    train(config)


if __name__ == "__main__":
    main()
