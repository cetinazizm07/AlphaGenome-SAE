"""Train one SAE on one tap of one fold.

Budget is counted in optimiser steps, not epochs. The pool size changes between
taps and folds, so an epoch is not a comparable unit; a step is.

The loop reports reconstruction health only: FVU, mean L0 and how much of the
dictionary ever fires. Checkpoints are selected on validation FVU. Concept
matching happens later, in concepts.py, so it cannot influence that choice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ag_sae import sae as sae_module
from ag_sae.vendor import sae_borzoi as vendor_module
from ag_sae.extract import ShardStore, sha256
from ag_sae.sae import BorzoiSAE, Recipe, build, channel_scale, save_inference_checkpoint


def _atomic_save(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.inference_mode()
def evaluate(model: BorzoiSAE, store: ShardStore, microbatch: int, device: str) -> dict[str, float]:
    """Streaming FVU, L0 and dictionary use over a whole split.

    FVU is the squared reconstruction error over the squared deviation from the
    split's own channel means. 1.0 means the model is no better than predicting
    those means.
    """
    model.eval()
    dim = store.dim
    total = torch.zeros(dim, dtype=torch.float64, device=device)
    total_sq = torch.zeros(dim, dtype=torch.float64, device=device)
    error = torch.zeros((), dtype=torch.float64, device=device)
    fired = torch.zeros(model.hidden, dtype=torch.int64, device=device)
    rows = 0
    for batch in store.batches(microbatch):
        raw = torch.from_numpy(batch).to(device)
        recon, codes, scaled = model(raw)
        if not torch.isfinite(recon).all() or not torch.isfinite(codes).all():
            raise ValueError("non-finite output during evaluation")
        target = scaled.double()
        error += (recon.double() - target).square().sum()
        total += target.sum(0)
        total_sq += target.square().sum(0)
        fired += (codes > 0).sum(0)
        rows += len(batch)
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


def train(
    train_store: ShardStore,
    val_store: ShardStore,
    recipe: Recipe,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    on_checkpoint: Callable[[Path], Any] | None = None,
    stop_after: int | None = None,
    log=print,
) -> dict:
    """Run the step budget, keeping the best checkpoint by validation FVU."""
    if train_store.dim != val_store.dim:
        raise ValueError("train and validation taps have different widths")
    recipe.validate(train_store.dim)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"

    identity = {"recipe": recipe.resolved(train_store.dim),
                "train": train_store.identity, "val": val_store.identity,
                "torch": str(torch.__version__), "numpy": np.__version__,
                "device_type": torch.device(device).type,
                # Hash the model definition too, not just this file. Editing the
                # SAE or the vendored core mid-campaign would otherwise resume
                # into a different model without complaint.
                "source": sha256(Path(__file__)),
                "sae_source": sha256(Path(sae_module.__file__)),
                "vendor_source": sha256(Path(vendor_module.__file__))}
    prior = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.exists() else None
    if prior is not None and prior["identity"] != identity:
        raise ValueError("data, recipe or source changed since this run started; use a new --out")

    if prior is not None:
        scale, fallback = prior["channel_scale"], prior["scale_fallback"]
    else:
        # Scales come from the training split only; using validation here would
        # leak its distribution into the model.
        scale, fallback = channel_scale(train_store.batches(recipe.microbatch), train_store.dim)

    model = build(train_store.dim, recipe, scale, device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=recipe.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    step, epoch, cursor, best = 0, 0, 0, float("inf")
    history: list[dict] = []
    order = None
    if prior is not None:
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        step, epoch, cursor, best = prior["step"], prior["epoch"], prior["cursor"], prior["best"]
        history = prior["history"]
        torch.set_rng_state(prior["rng"].cpu())
        log(f"resuming at step {step}, epoch {epoch}, row {cursor}")

    def persist(is_best: bool = False) -> None:
        payload = {"identity": identity, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                   "cursor": cursor, "best": best, "history": history,
                   "channel_scale": scale, "scale_fallback": fallback,
                   "rng": torch.get_rng_state()}
        if is_best:
            save_inference_checkpoint(out_dir / "best.pt", model, recipe,
                                      extra={"step": step, "history": history})
            if on_checkpoint:
                on_checkpoint(out_dir / "best.pt")
        _atomic_save(latest_path, payload)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    started = time.monotonic()
    accumulation = recipe.batch_tokens // recipe.microbatch
    while step < recipe.steps:
        if order is None:
            order = train_store.order(recipe.seed, epoch)
        if cursor + recipe.batch_tokens > len(order):
            epoch += 1
            cursor, order = 0, None
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_error = torch.zeros((), device=device)
        for chunk in range(accumulation):
            lo = cursor + chunk * recipe.microbatch
            rows = order[lo:lo + recipe.microbatch]
            raw = torch.from_numpy(train_store.take(rows)).to(device)
            recon, _, scaled = model(raw)
            error = (recon - scaled).square().sum()
            # Divide by the whole step's element count so accumulation gives
            # the same gradient as one large batch.
            (error / (recipe.batch_tokens * train_store.dim)).backward()
            batch_error += error.detach()
        if not torch.isfinite(batch_error):
            raise ValueError(f"non-finite training loss at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        cursor += recipe.batch_tokens
        step += 1

        if step % recipe.eval_every == 0 or step == recipe.steps:
            metrics = evaluate(model, val_store, recipe.microbatch, device)
            improved = metrics["fvu"] < best
            best = min(best, metrics["fvu"])
            history.append({"step": step, "epoch": epoch,
                            "train_mse": float(batch_error / (recipe.batch_tokens * train_store.dim)),
                            "validation": metrics})
            log(f"step {step}/{recipe.steps} val_fvu={metrics['fvu']:.5f} "
                f"L0={metrics['mean_l0']:.1f} dict_used={metrics['dictionary_used']:.1%}")
            persist(is_best=improved)
        elif step % recipe.checkpoint_every == 0:
            persist()
        if stop_after is not None and step >= stop_after:
            persist()
            return {"status": "interrupted", "step": step}

    persist()
    result = {"status": "complete", "steps": step, "epochs": epoch,
              "best_validation_fvu": best,
              "scale_fallback_channels": fallback,
              "seconds": time.monotonic() - started,
              "recipe": recipe.resolved(train_store.dim),
              "tap": train_store.tap,
              "best_checkpoint": str(out_dir / "best.pt")}
    (out_dir / "training.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    import argparse
    from dataclasses import fields

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", required=True, help="extraction directory")
    parser.add_argument("--tap", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    for field in fields(Recipe):
        parser.add_argument("--" + field.name.replace("_", "-"),
                            type=type(field.default), default=field.default)
    args = parser.parse_args(argv)

    recipe = Recipe(**{f.name: getattr(args, f.name) for f in fields(Recipe)})
    train_store = ShardStore(args.activations, args.tap, "train")
    val_store = ShardStore(args.activations, args.tap, "val")
    print(f"tap {args.tap}: {len(train_store):,} train rows, {len(val_store):,} val rows, "
          f"dict {recipe.hidden(train_store.dim)}, k {recipe.k(train_store.dim)}")
    result = train(train_store, val_store, recipe, args.out, device=args.device)
    print(json.dumps({k: v for k, v in result.items() if k != "recipe"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
