"""TopK sparse autoencoder: the published core plus the bits it needs here.

The encoder/decoder maths lives in `vendor/sae_borzoi.py`, copied unmodified
from calico/sae-borzoi. This file adds only what is specific to AlphaGenome:
per-channel scaling computed from the training split, and a checkpoint format
that `concepts.py` can load.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ag_sae.vendor.sae_borzoi import SparseAutoencoder


@dataclass(frozen=True)
class Recipe:
    """Training settings.

    expansion, topk_pct, lr and the MSE-only loss are the published sae-borzoi
    values. `steps` is not: the reference publishes the model, not its training
    loop, so the budget is ours to choose and to justify.
    """

    expansion: int = 4
    topk_pct: float = 0.05
    #: "dictionary" applies the percentage to the expanded latents (the paper);
    #: "input" applies it to the input channels (the released script).
    topk_basis: str = "dictionary"
    lr: float = 1e-5
    #: Optimiser steps, not epochs. Epoch count depends on pool size, so it is
    #: the wrong unit once the pool changes.
    steps: int = 3000
    batch_tokens: int = 16384
    microbatch: int = 1024
    checkpoint_every: int = 100
    eval_every: int = 250
    seed: int = 0
    #: Fixed dictionary width. Set this to compare taps of different widths on
    #: equal terms; 0 means derive it from `expansion`.
    n_features: int = 0

    def hidden(self, d_in: int) -> int:
        return self.n_features or self.expansion * d_in

    def k(self, d_in: int) -> int:
        basis = d_in if self.topk_basis == "input" else self.hidden(d_in)
        return int(self.topk_pct * basis)

    def validate(self, d_in: int) -> None:
        for name in ("expansion", "steps", "batch_tokens", "microbatch",
                     "checkpoint_every", "eval_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.topk_basis not in ("input", "dictionary"):
            raise ValueError("topk_basis must be 'input' or 'dictionary'")
        if not 0 < self.topk_pct <= 1 or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("invalid sparsity or learning rate")
        if self.microbatch > self.batch_tokens:
            raise ValueError("microbatch cannot exceed batch_tokens")
        if self.batch_tokens % self.microbatch:
            raise ValueError("batch_tokens must be a multiple of microbatch")
        if not 1 <= self.k(d_in) <= self.hidden(d_in):
            raise ValueError(f"k={self.k(d_in)} outside 1..{self.hidden(d_in)}")
        if self.n_features < 0:
            raise ValueError("n_features cannot be negative")

    def resolved(self, d_in: int) -> dict[str, Any]:
        return {**asdict(self), "d_in": d_in, "hidden": self.hidden(d_in),
                "k": self.k(d_in), "token_layernorm": True,
                "optimizer": "Adam", "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0, "loss": "MSE", "lr_schedule": "constant"}


def channel_scale(batches, d_in: int) -> tuple[np.ndarray, int]:
    """Per-channel divisor, from the training split only.

    Upstream divides by each channel's maximum. AlphaGenome embeddings are
    signed, so a channel can have a non-positive maximum; those fall back to
    abs-max, and all-zero channels to 1. Returns the scale and how many
    channels needed the fallback.
    """
    highest = np.full(d_in, -np.inf, dtype=np.float64)
    largest_abs = np.zeros(d_in, dtype=np.float64)
    for batch in batches:
        array = np.asarray(batch, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("non-finite activations while computing channel scale")
        highest = np.maximum(highest, array.max(0))
        largest_abs = np.maximum(largest_abs, np.abs(array).max(0))
    fallback = highest <= 1e-8
    scale = np.where(fallback, np.where(largest_abs > 1e-8, largest_abs, 1.0), highest)
    return scale.astype(np.float32), int(fallback.sum())


class BorzoiSAE(nn.Module):
    """The published SAE, wrapped with AlphaGenome's channel scaling."""

    def __init__(self, d_in: int, hidden: int, k: int, scale) -> None:
        super().__init__()
        if d_in < 2 or not 1 <= k <= hidden:
            raise ValueError("invalid SAE dimensions")
        scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
        if scale.shape != (d_in,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("channel scale must be positive, finite, one per channel")
        self.register_buffer("channel_scale", scale)
        self.core = SparseAutoencoder(d_in, hidden, k, sparsity_method="topk_o",
                                      normalize=True)
        # Only smooth_topk uses temperature, and we do not use smooth_topk.
        self.core.temperature.requires_grad_(False)
        self.d_in, self.hidden, self.k = d_in, hidden, k

    def forward(self, raw: torch.Tensor):
        """Return (reconstruction, sparse codes, scaled input).

        The loss is taken in scaled space, as in the upstream training loop.
        """
        scaled = raw / self.channel_scale
        pre, params = self.core.encode(scaled)
        codes = self.core.get_sparse_activations(self.core.activation(pre))
        return self.core.decode(codes, params), codes, scaled

    @torch.no_grad()
    def reconstruct(self, raw: torch.Tensor):
        """Reconstruction back in raw units, plus the codes."""
        recon, codes, _ = self(raw)
        return recon * self.channel_scale, codes

    def inference_state(self) -> dict[str, torch.Tensor]:
        """The tensors `concepts.FrozenSAE` expects, with flat key names."""
        return {"encoder.weight": self.core.encoder.weight.detach().cpu().clone(),
                "decoder.weight": self.core.decoder.weight.detach().cpu().clone(),
                "latent_bias": self.core.latent_bias.detach().cpu().clone(),
                "pre_bias": self.core.pre_bias.detach().cpu().clone(),
                "channel_scale": self.channel_scale.detach().cpu().clone()}


def save_inference_checkpoint(path: str | Path, model: BorzoiSAE, recipe: Recipe,
                              extra: dict[str, Any] | None = None) -> Path:
    """Write the small file used for analysis. Resume state goes elsewhere."""
    state = model.inference_state()
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("refusing to save non-finite parameters")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": state, "recipe": recipe.resolved(model.d_in),
                **(extra or {})}, path)
    return path


def build(d_in: int, recipe: Recipe, scale, device: str = "cpu") -> BorzoiSAE:
    """Create a model on `device` with the recipe's dimensions."""
    recipe.validate(d_in)
    torch.manual_seed(recipe.seed)
    return BorzoiSAE(d_in, recipe.hidden(d_in), recipe.k(d_in), scale).to(device)
