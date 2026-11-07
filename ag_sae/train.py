"""Train SAEs on one tap of one fold.

Budget is counted in optimiser steps, not epochs. The pool size changes between
taps and folds, so an epoch is not a comparable unit; a step is.

The loop reports reconstruction health only: FVU, mean L0, how much of the
dictionary fires on validation and how many latents are dead by the training
definition. Checkpoints are selected on validation FVU. Concept matching
happens later, in concepts.py, so it cannot influence that choice.

The split is loaded once into an `ActivationPool` (on the GPU when it fits),
so several seeds of one tap can be trained in one process without reading the
shards again: ``--seeds 0 1 2 --out 'sae/fold3/resid_pre_b8_seed{seed}'``.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ag_sae import pool as pool_module
from ag_sae import sae as sae_module
from ag_sae.vendor import sae_borzoi as vendor_module
from ag_sae.extract import ShardStore, sha256
from ag_sae.pool import ActivationPool
from ag_sae.sae import (BorzoiSAE, Recipe, build, channel_scale, derive_seed,
                        save_inference_checkpoint)

PRECISIONS = ("fp32", "tf32", "bf16")


def _atomic_save(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def configure_precision(precision: str) -> None:
    """fp32: exact float32 matmuls. tf32: tensor-core float32 (A100/H100).
    bf16: tf32 plus bfloat16 autocast for the SAE's matmuls."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"


def _autocast(device: str, precision: str):
    if precision == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def evaluate(model: BorzoiSAE, data, microbatch: int, device: str) -> dict[str, float]:
    """Streaming FVU, L0 and dictionary use over a whole split.

    `data` is a ShardStore or an ActivationPool. FVU is the squared
    reconstruction error over the squared deviation from the split's own
    channel means. 1.0 means the model is no better than predicting those
    means.
    """
    model.eval()
    dim = data.dim
    # MPS has no float64; it is only used for local smoke tests.
    wide = torch.float32 if torch.device(device).type == "mps" else torch.float64
    total = torch.zeros(dim, dtype=wide, device=device)
    total_sq = torch.zeros(dim, dtype=wide, device=device)
    error = torch.zeros((), dtype=wide, device=device)
    fired = torch.zeros(model.hidden, dtype=torch.int64, device=device)
    rows = 0
    for batch in data.batches(microbatch):
        raw = torch.as_tensor(batch).to(device)
        recon, codes, scaled = model(raw)
        if not torch.isfinite(recon).all() or not torch.isfinite(codes).all():
            raise ValueError("non-finite output during evaluation")
        target = scaled.to(wide)
        error += (recon.to(wide) - target).square().sum()
        total += target.sum(0)
        total_sq += target.square().sum(0)
        fired += (codes > 0).sum(0)
        rows += len(raw)
    variance = (total_sq - total.square() / rows).sum()
    if not torch.isfinite(variance) or variance <= 0:
        raise ValueError("split has no variance to explain")
    counts = fired.cpu().numpy()
    return {"rows": rows,
            "fvu": float(error / variance),
            "mse": float(error / (rows * dim)),
            "mean_l0": float(counts.sum() / rows),
            "features_fired": int((counts > 0).sum()),
            "features_never_fired": int((counts == 0).sum()),
            "dictionary_used": float((counts > 0).mean())}


def _as_pool(data, device: str, log) -> ActivationPool:
    return data if isinstance(data, ActivationPool) else ActivationPool(data, device, log=log)


def train(
    train_data: ShardStore | ActivationPool,
    val_data: ShardStore | ActivationPool,
    recipe: Recipe,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    precision: str = "tf32",
    on_checkpoint: Callable[[Path], Any] | None = None,
    stop_after: int | None = None,
    log=print,
) -> dict:
    """Run the step budget, keeping the best checkpoint by validation FVU."""
    train_pool, val_pool = _as_pool(train_data, device, log), _as_pool(val_data, device, log)
    if train_pool.dim != val_pool.dim:
        raise ValueError("train and validation taps have different widths")
    dim = train_pool.dim
    recipe.validate(dim)
    configure_precision(precision)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"

    # k is part of the key so k~409 and k=64 runs with one --seed are
    # independent draws, not a shared initialisation.
    data_key = f'{train_pool.identity["index"]}|k={recipe.k(dim)}'

    init_seed = derive_seed(recipe, train_pool.tap, data_key, "init")
    data_seed = derive_seed(recipe, train_pool.tap, data_key, "data")
    identity = {"recipe": recipe.resolved(dim),
                "train": train_pool.identity, "val": val_pool.identity,
                "init_seed": init_seed, "data_seed": data_seed,
                "precision": precision,
                "torch": str(torch.__version__), "numpy": np.__version__,
                "device_type": torch.device(device).type,
                # Hash the model definition too, not just this file. Editing the
                # SAE or the vendored core mid-campaign would otherwise resume
                # into a different model without complaint.
                "source": sha256(Path(__file__)),
                "sae_source": sha256(Path(sae_module.__file__)),
                "pool_source": sha256(Path(pool_module.__file__)),
                "vendor_source": sha256(Path(vendor_module.__file__))}
    prior = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("data, recipe or source changed since this run started; use a new --out")

    if prior is not None:
        scale, fallback = prior["channel_scale"], prior["scale_fallback"]
    else:
        # Scales come from the training split only; using validation here would
        # leak its distribution into the model.
        scale, fallback = channel_scale(train_pool.batches(65536), dim)

    model = build(dim, recipe, scale, device, seed=init_seed)
    if prior is None and recipe.pre_bias_init == "mean":
        model.init_pre_bias_(train_pool.sample(65536, init_seed))
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=recipe.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    # Training tokens since each latent last fired; >= dead_tokens means dead.
    since_fired = torch.zeros(model.hidden, dtype=torch.int64, device=device)
    step, epoch, cursor, best = 0, 0, 0, float("inf")
    history: list[dict] = []
    order = None
    if prior is not None:
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        since_fired.copy_(prior["since_fired"].to(device))
        step, epoch, cursor, best = prior["step"], prior["epoch"], prior["cursor"], prior["best"]
        history = prior["history"]
        torch.set_rng_state(prior["rng"].cpu())
        log(f"resuming at step {step}, epoch {epoch}, row {cursor}")

    def persist(is_best: bool = False) -> None:
        payload = {"identity": identity, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                   "cursor": cursor, "best": best, "history": history,
                   "since_fired": since_fired.cpu(),
                   "channel_scale": scale, "scale_fallback": fallback,
                   "rng": torch.get_rng_state()}
        if is_best:
            save_inference_checkpoint(out_dir / "best.pt", model, recipe,
                                      extra={"step": step, "history": history})
            if on_checkpoint:
                on_checkpoint(out_dir / "best.pt")
        _atomic_save(latest_path, payload)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    norm = float(recipe.batch_tokens * dim)
    started, window_started, window_steps = time.monotonic(), time.monotonic(), 0
    while step < recipe.steps:
        if order is None:
            order = train_pool.order(data_seed, epoch, recipe.shuffle)
        if cursor + recipe.batch_tokens > len(order):
            epoch += 1
            cursor, order = 0, None
            continue

        lr = recipe.learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        dead = since_fired >= recipe.dead_tokens if recipe.uses_aux else None
        batch_error = torch.zeros((), device=device)
        batch_aux = torch.zeros((), device=device)
        fired = torch.zeros(model.hidden, dtype=torch.bool, device=device)
        rows = order[cursor:cursor + recipe.batch_tokens]
        for chunk in rows.split(recipe.microbatch):
            raw = train_pool.take(chunk)
            with _autocast(device, precision):
                recon, codes, scaled, aux = model.forward_train(raw, dead, recipe.aux_k)
            recon = recon.float()
            error = (recon - scaled).square().sum()
            # Divide by the whole step's element count so accumulation gives
            # the same gradient as one large batch.
            loss = error / norm
            if aux is not None:
                residual = (scaled - recon).detach()
                aux_error = (aux.float() - residual).square().sum()
                loss = loss + recipe.aux_coef * aux_error / norm
                batch_aux += aux_error.detach()
            loss.backward()
            batch_error += error.detach()
            fired |= (codes > 0).any(0)
        if not torch.isfinite(batch_error):
            raise ValueError(f"non-finite training loss at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        if recipe.decoder_norm == "unit":
            model.project_decoder_grad_()
        optimizer.step()
        if recipe.decoder_norm == "unit":
            model.normalize_decoder_()
        since_fired += recipe.batch_tokens
        since_fired[fired] = 0
        cursor += recipe.batch_tokens
        step += 1
        window_steps += 1

        if step % recipe.eval_every == 0 or step == recipe.steps:
            metrics = evaluate(model, val_pool, max(recipe.microbatch, 8192), device)
            improved = metrics["fvu"] < best
            best = min(best, metrics["fvu"])
            elapsed = time.monotonic() - window_started
            n_dead = int((since_fired >= recipe.dead_tokens).sum())
            history.append({"step": step, "epoch": epoch, "lr": lr,
                            "train_mse": float(batch_error / norm),
                            "train_aux_mse": float(batch_aux / norm),
                            "dead_latents": n_dead,
                            "validation": metrics})
            log(f"step {step}/{recipe.steps} lr={lr:.2e} val_fvu={metrics['fvu']:.5f} "
                f"L0={metrics['mean_l0']:.1f} dict_used={metrics['dictionary_used']:.1%} "
                f"dead={n_dead} ({window_steps * recipe.batch_tokens / max(elapsed, 1e-9):,.0f} tok/s)")
            persist(is_best=improved)
            window_started, window_steps = time.monotonic(), 0
        elif step % recipe.checkpoint_every == 0:
            persist()
        if stop_after is not None and step >= stop_after:
            persist()
            return {"status": "interrupted", "step": step}

    persist()
    result = {"status": "complete", "steps": step, "epochs": epoch,
              "best_validation_fvu": best,
              "scale_fallback_channels": fallback,
              "dead_latents": int((since_fired >= recipe.dead_tokens).sum()),
              "seconds": time.monotonic() - started,
              "recipe": recipe.resolved(dim),
              "init_seed": init_seed, "data_seed": data_seed, "precision": precision,
              "tap": train_pool.tap,
              "best_checkpoint": str(out_dir / "best.pt")}
    (out_dir / "training.json").write_text(json.dumps(result, indent=2))
    return result


def _is_complete(out_dir: Path) -> bool:
    path = out_dir / "training.json"
    return path.is_file() and json.loads(path.read_text()).get("status") == "complete"


def main(argv=None) -> int:
    import argparse
    from dataclasses import fields, replace

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", required=True, help="extraction directory")
    parser.add_argument("--tap", required=True)
    parser.add_argument("--out", required=True,
                        help="output directory; with --seeds it must contain {seed}")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="train one SAE per seed on the same in-memory pool")
    parser.add_argument("--recipe", choices=("v2", "legacy"), default="v2",
                        help="base recipe; any flag below overrides it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=PRECISIONS, default="tf32")
    parser.add_argument("--placement", choices=("auto", "gpu", "host"), default="auto",
                        help="where the training split lives")
    defaults = Recipe()
    for field in fields(Recipe):
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(getattr(defaults, field.name)), default=None)
    args = parser.parse_args(argv)

    base = Recipe() if args.recipe == "v2" else Recipe.legacy()
    overrides = {f.name: getattr(args, f.name) for f in fields(Recipe)
                 if getattr(args, f.name) is not None}
    base = replace(base, **overrides)
    seeds = args.seeds if args.seeds else [base.seed]
    if len(seeds) > 1 and "{seed}" not in args.out:
        parser.error("--out must contain {seed} when training several seeds")

    train_store = ShardStore(args.activations, args.tap, "train")
    val_store = ShardStore(args.activations, args.tap, "val")
    print(f"tap {args.tap}: {len(train_store):,} train rows, {len(val_store):,} val rows, "
          f"dict {base.hidden(train_store.dim)}, k {base.k(train_store.dim)}, recipe {args.recipe}")
    jobs = [(replace(base, seed=s), Path(args.out.format(seed=s))) for s in seeds]
    jobs = [(r, out) for r, out in jobs if not _is_complete(out)]
    if not jobs:
        print("all requested runs are complete")
        return 0
    train_pool = ActivationPool(train_store, args.device, args.placement)
    val_pool = ActivationPool(val_store, args.device, args.placement)
    for recipe, out in jobs:
        print(f"--- seed {recipe.seed} -> {out}")
        result = train(train_pool, val_pool, recipe, out, device=args.device,
                       precision=args.precision)
        print(json.dumps({k: v for k, v in result.items() if k != "recipe"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
