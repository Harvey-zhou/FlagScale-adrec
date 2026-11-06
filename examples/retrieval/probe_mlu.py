"""Cambricon MLU stack probe for the retrieval runs.

It answers, in one pass and before any model is loaded, the questions the
retrieval code cannot answer for itself on torch_mlu:

* Does ``torch.autocast("mlu", dtype=torch.bfloat16)`` really produce bf16?
  BGE-M3 and CLIP get their bf16 compute *only* from autocast, and
  ``autocast_context`` in train_retrieval.py swallows RuntimeError/TypeError, so
  an unsupported dtype would give a pure fp32 run with no error at all.
* Is a float64 ``all_reduce`` exact?  ``reduce_metrics`` reduces the reported
  loss in float64.  The public torch_mlu maps ``kDouble`` to ``cnclFloat``; if
  that is still true the reported loss is wrong without any error.
* Do the operators FSDP2 calls internally run on the card, with CPU fallback
  off (``_chunk_cat``, ``split_with_sizes_copy.out``, ``_foreach_*``)?
* Does a small FSDP2 model train and gather back to the right weights on the
  requested number of cards, through the same ``apply_fsdp2`` the reranker uses?

Single card:   python examples/retrieval/probe_mlu.py
Several cards: python -m torch.distributed.run --nproc_per_node=2 examples/retrieval/probe_mlu.py

``examples/retrieval/run-cambricon.sh`` calls it with the right form.  Nothing
here is a training result; it only reports what the stack can and cannot do.
Exit status is non-zero when a check marked FAIL did not pass.
"""

from __future__ import annotations

import datetime
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import through the platform adapter: it refuses to import torch_mlu when
# another backend owns PrivateUse1, and it never uses gpu_migration.
from flagscale.platforms.platform_register import PLATFORMS

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    if RANK == 0:
        print(f"[{status:4}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def check(name: str, fn, critical: bool = True) -> None:
    """Run ``fn``; it returns a detail string, or raises on failure."""
    try:
        detail = fn()
        record("PASS", name, detail or "")
    except Exception as exc:  # the type and message are the evidence
        record("FAIL" if critical else "WARN", name, f"{type(exc).__name__}: {str(exc)[:300]}")


def main() -> int:
    if "mlu" not in PLATFORMS:
        print("FAIL: the mlu platform is not available (torch_mlu import or card visibility).")
        print(f"      registered platforms: {sorted(PLATFORMS)}")
        return 1
    platform = PLATFORMS["mlu"]
    platform.set_device(LOCAL_RANK)
    dev = platform.device(LOCAL_RANK)

    # ---- versions and visible cards ------------------------------------------------------
    import torch_mlu

    if RANK == 0:
        print(f"torch {torch.__version__}  torch_mlu {getattr(torch_mlu, '__version__', '?')}")
        print(f"ENABLE_FALLBACK_TO_CPU={os.environ.get('ENABLE_FALLBACK_TO_CPU', '(unset)')} "
              f"MLU_VISIBLE_DEVICES={os.environ.get('MLU_VISIBLE_DEVICES', '(unset)')} "
              f"PYTORCH_MLU_ALLOC_CONF={os.environ.get('PYTORCH_MLU_ALLOC_CONF', '(unset)')}")
        print(f"privateuse1 backend name: {torch._C._get_privateuse1_backend_name()}  "
              f"cards visible: {torch.mlu.device_count()}  world size: {WORLD}")
    if os.environ.get("ENABLE_FALLBACK_TO_CPU", "") != "0" and os.environ.get("ALLOW_CPU_FALLBACK", "") != "1":
        record("FAIL", "CPU fallback off", "ENABLE_FALLBACK_TO_CPU is not 0; a missing kernel would run on the CPU silently")
    def cards_match() -> str:
        if torch.mlu.device_count() != WORLD:
            raise RuntimeError(f"{torch.mlu.device_count()} cards visible, world size {WORLD}; "
                               "MLU_VISIBLE_DEVICES did not select the requested cards")
        return str(WORLD)

    check("cards visible == world size", cards_match)

    # ---- autocast: the silent-loss risk ----------------------------------------------------
    def autocast_bf16() -> str:
        a = torch.randn(64, 64, device=dev)
        b = torch.randn(64, 64, device=dev)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.autocast("mlu", dtype=torch.bfloat16):
                outputs = {
                    "matmul": a @ b,
                    "linear": F.linear(a, b),
                    "bmm": torch.bmm(a[None], b[None]),
                }
        notes = "; ".join(str(w.message)[:90] for w in caught)
        bad = {k: str(v.dtype) for k, v in outputs.items() if v.dtype != torch.bfloat16}
        if bad:
            raise RuntimeError(f"autocast did not give bf16: {bad}; warnings: {notes or 'none'}")
        return "matmul, linear and bmm are bf16 under autocast" + (f" (warnings: {notes})" if notes else "")

    check("autocast bf16 produces bf16", autocast_bf16)
    def bf16_supported() -> str:
        if not torch.mlu.is_bf16_supported():
            raise RuntimeError("torch.mlu.is_bf16_supported() is False")
        return "True"

    check("torch.mlu.is_bf16_supported()", bf16_supported)

    def bf16_accuracy() -> str:
        gen = torch.Generator().manual_seed(0)
        a = torch.randn(256, 512, generator=gen)
        b = torch.randn(512, 128, generator=gen)
        ref = a @ b
        out = (a.to(dev, torch.bfloat16) @ b.to(dev, torch.bfloat16)).float().cpu()
        rel = float((out - ref).norm() / ref.norm())
        if rel > 2e-2:
            raise RuntimeError(f"bf16 matmul relative error {rel:.3g} > 2e-2")
        return f"bf16 matmul relative error {rel:.2e}"

    check("bf16 matmul accuracy vs fp32 CPU", bf16_accuracy)

    def sdpa_mask() -> str:
        q = torch.randn(2, 4, 32, 64, device=dev, dtype=torch.bfloat16)
        mask = torch.ones(2, 1, 32, 32, device=dev, dtype=torch.bool).tril()
        out = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
        ref = F.scaled_dot_product_attention(q.float().cpu(), q.float().cpu(), q.float().cpu(),
                                             attn_mask=mask.cpu())
        err = float((out.float().cpu() - ref).abs().max())
        if err > 5e-2:
            raise RuntimeError(f"max abs error {err:.3g}")
        return f"bf16 SDPA with a bool mask, max abs error {err:.2e}"

    check("SDPA bf16 with attention mask", sdpa_mask)

    # ---- the runtime pieces FSDP2 and the training loop touch -----------------------------
    def runtime_apis() -> str:
        missing = [n for n in ("Stream", "Event", "current_stream", "stream", "synchronize", "empty_cache",
                               "memory_allocated", "memory_reserved", "max_memory_allocated",
                               "reset_peak_memory_stats", "manual_seed_all", "mem_get_info")
                   if not hasattr(torch.mlu, n)]
        if missing:
            raise RuntimeError(f"torch.mlu lacks {missing}")
        s = torch.mlu.Stream()
        e = torch.mlu.Event()
        with torch.mlu.stream(s):
            x = torch.randn(8, device=dev) * 2
            e.record(s)
        e.synchronize()
        torch.mlu.current_stream().wait_stream(s)
        stats = platform.memory_stats()
        platform.empty_cache()
        return f"stream/event ok; allocated {stats['allocated'] / 2**20:.0f} MiB"

    check("torch.mlu runtime APIs", runtime_apis)

    def pinned() -> str:
        host = torch.zeros(1024).pin_memory()
        if not host.is_pinned():
            raise RuntimeError("pin_memory() returned an unpinned tensor")
        out = host.to(dev, non_blocking=True)
        torch.mlu.synchronize()
        return f"pinned {out.shape[0]} floats copied to the card"

    check("pinned host memory", pinned)

    # ---- operators FSDP2 calls inside its collectives --------------------------------------
    def fsdp_ops() -> str:
        # chunk_cat: pack per-rank shards the way reduce-scatter input is built.
        tensors = [torch.randn(6, 5), torch.randn(4, 5)]
        ref = torch._chunk_cat(tensors, dim=0, num_chunks=2)
        out = torch._chunk_cat([t.to(dev) for t in tensors], dim=0, num_chunks=2).cpu()
        if not torch.equal(ref, out):
            raise RuntimeError("_chunk_cat differs from the CPU result")
        # split_with_sizes_copy.out: unpack the all-gathered buffer.
        flat = torch.randn(20, device=dev)
        outs = [torch.empty(8, device=dev), torch.empty(12, device=dev)]
        torch.split_with_sizes_copy(flat, [8, 12], dim=0, out=outs)
        if not (torch.equal(outs[0].cpu(), flat[:8].cpu()) and torch.equal(outs[1].cpu(), flat[8:].cpu())):
            raise RuntimeError("split_with_sizes_copy.out differs from a slice")
        # _foreach_copy_ / _foreach_norm: used by the copy-in path and by clip_grad_norm_.
        a = [torch.randn(7, device=dev), torch.randn(3, 4, device=dev)]
        b = [torch.zeros_like(x) for x in a]
        torch._foreach_copy_(b, a)
        if not all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(a, b)):
            raise RuntimeError("_foreach_copy_ result differs")
        norms = torch._foreach_norm(a)
        ref_norms = [x.norm() for x in a]
        if not all(torch.allclose(n.cpu(), r.cpu(), rtol=1e-5) for n, r in zip(norms, ref_norms)):
            raise RuntimeError("_foreach_norm differs from per-tensor norm")
        # The custom ops torch registers for FSDP2 on PrivateUse1.
        names = ["all_gather_copy_in", "split_with_sizes_copy", "chunk_cat"]
        absent = [n for n in names if not hasattr(torch.ops.fsdp, n)]
        if absent:
            raise RuntimeError(f"torch.ops.fsdp lacks {absent}")
        return "chunk_cat, split_with_sizes_copy.out, foreach_copy_, foreach_norm and torch.ops.fsdp.* ok"

    check("FSDP2 internal operators (strict, no CPU fallback)", fsdp_ops)

    # ---- distributed -----------------------------------------------------------------------
    if WORLD > 1:
        import torch.distributed as dist

        # A short timeout: a rank that fails a check must not leave the others
        # waiting in a collective for the default 10 minutes.
        dist.init_process_group(backend=platform.dist_backend(), init_method="env://",
                                timeout=datetime.timedelta(minutes=3))
        probe_distributed(dist, dev, platform)
        dist.barrier()
        dist.destroy_process_group()
    elif RANK == 0:
        record("INFO", "distributed checks skipped", "run with --nproc_per_node=2 to check CNCL and FSDP2")

    # ---- summary ---------------------------------------------------------------------------
    failures = [r for r in _results if r[0] == "FAIL"]
    if RANK == 0:
        print(f"\nsummary: {sum(r[0] == 'PASS' for r in _results)} pass, "
              f"{sum(r[0] == 'WARN' for r in _results)} warn, {len(failures)} fail")
    return 1 if failures else 0


def probe_distributed(dist, dev, platform) -> None:
    def fp64_all_reduce() -> str:
        # 1 + 2**-40 is not representable in fp32.  If CNCL moved the data as
        # fp32 the sum below would lose the 2**-40 term; if it moved 8 bytes as
        # 4-byte floats the values would be garbage.
        eps = 2.0**-40
        x = torch.tensor([1.5, 1.0 + eps], dtype=torch.float64, device=dev)
        dist.all_reduce(x)
        got = x.cpu().tolist()
        want = [1.5 * WORLD, WORLD * (1.0 + eps)]
        if abs(got[0] - want[0]) > 1e-12 or abs(got[1] - want[1]) > WORLD * eps / 2:
            raise RuntimeError(f"float64 all_reduce gave {got}, expected {want}; "
                               "reduce_metrics' reported loss would be wrong")
        return f"exact across {WORLD} ranks"

    check("float64 all_reduce is exact (reduce_metrics)", fp64_all_reduce)

    def collectives() -> str:
        x = torch.full((4,), float(RANK + 1), device=dev, dtype=torch.bfloat16)
        dist.all_reduce(x)
        if not torch.allclose(x.float().cpu(), torch.full((4,), float(sum(range(1, WORLD + 1))))):
            raise RuntimeError(f"bf16 all_reduce gave {x.tolist()}")
        shard = torch.full((3,), float(RANK), device=dev)
        gathered = torch.empty(3 * WORLD, device=dev)
        dist.all_gather_into_tensor(gathered, shard)
        want = torch.arange(WORLD, dtype=torch.float32).repeat_interleave(3)
        if not torch.equal(gathered.cpu(), want):
            raise RuntimeError(f"all_gather_into_tensor gave {gathered.tolist()}")
        full = torch.arange(WORLD * 3, dtype=torch.float32, device=dev)
        scattered = torch.empty(3, device=dev)
        dist.reduce_scatter_tensor(scattered, full)
        want = torch.arange(RANK * 3, RANK * 3 + 3, dtype=torch.float32) * WORLD
        if not torch.equal(scattered.cpu(), want):
            raise RuntimeError(f"reduce_scatter_tensor gave {scattered.tolist()}")
        return "bf16 all_reduce, all_gather_into_tensor, reduce_scatter_tensor"

    check("CNCL collectives", collectives)

    def coalesced_methods() -> str:
        backend = dist.distributed_c10d._get_default_group()._get_backend(dev)
        names = ("allgather_into_tensor_coalesced", "reduce_scatter_tensor_coalesced",
                 "_allgather_base", "_reduce_scatter_base")
        present = {n: hasattr(backend, n) for n in names}
        missing = [n for n, ok in present.items() if not ok]
        if missing:
            raise RuntimeError(f"{type(backend).__name__} lacks {missing}; "
                               "install_coalesced_allgather_shim must cover this class")
        return f"{type(backend).__name__}: all present, so the shim should not be needed"

    check("ProcessGroup coalesced methods", coalesced_methods, critical=False)

    check("tiny FSDP2 model trains and gathers back", lambda: fsdp2_roundtrip(dist, dev))


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(self.norm(x))))


class _Tiny(nn.Module):
    def __init__(self, vocab: int = 64, width: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(_Block(width) for _ in range(depth))
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def fsdp2_roundtrip(dist, dev) -> str:
    """Shard a small model with the reranker's own apply_fsdp2 and check it."""
    from flagscale.train.utils import parallel as parallel_utils

    if not parallel_utils.fsdp2_available():
        raise RuntimeError("composable FSDP2 is not exposed by this torch")

    torch.manual_seed(1234)
    reference = _Tiny()
    initial = {k: v.clone() for k, v in reference.state_dict().items()}
    ids = torch.randint(0, 64, (4, 16), generator=torch.Generator().manual_seed(7))
    target = torch.randint(0, 64, (4, 16), generator=torch.Generator().manual_seed(8))

    # fp32 CPU reference loss for the first step.
    ref_loss = F.cross_entropy(reference(ids).view(-1, 64), target.view(-1)).item()

    model = _Tiny()
    model.load_state_dict(initial)
    # The reranker builds the model on the CPU, shards it, then moves the local
    # shards to the card in the master dtype; do the same.
    parallel_utils.apply_fsdp2(model, device_type="mlu", world_size=WORLD,
                               compute_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    model.to(device=dev, dtype=torch.float32)

    gathered = parallel_utils.full_state_dict(model)
    for key, value in initial.items():
        if key not in gathered:
            raise RuntimeError(f"full_state_dict lost {key}")
        if not torch.equal(gathered[key].float().cpu(), value):
            raise RuntimeError(f"full_state_dict changed {key} before any update "
                               "(this is how the earlier FSDP2 save bugs looked)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(ids.to(dev)).float().view(-1, 64), target.to(dev).view(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if abs(losses[0] - ref_loss) > 0.05 * max(abs(ref_loss), 1.0):
        raise RuntimeError(f"first FSDP2 loss {losses[0]:.4f} vs fp32 CPU {ref_loss:.4f}")
    if not losses[-1] < losses[0]:
        raise RuntimeError(f"loss did not fall over 3 steps: {losses}")
    after = parallel_utils.full_state_dict(model)
    if all(torch.equal(after[k].float().cpu(), initial[k]) for k in initial):
        raise RuntimeError("weights did not change after 3 optimizer steps")
    return f"loss {losses[0]:.4f} (fp32 CPU {ref_loss:.4f}) -> {losses[-1]:.4f}; full_state_dict round trip exact"


if __name__ == "__main__":
    sys.exit(main())
