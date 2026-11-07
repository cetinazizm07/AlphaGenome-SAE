"""TopK sparse autoencoder: the published core plus the bits it needs here.

The encoder/decoder maths lives in `vendor/sae_borzoi.py`, copied unmodified
from calico/sae-borzoi. This file adds only what is specific to AlphaGenome:
per-channel scaling computed from the training split, a checkpoint format that
`concepts.py` can load, and the training-time pieces the published recipe
lacks (see `Recipe`).

Two recipes are supported:

``Recipe()`` (v2, the default)
    Adds the standard defences against dead latents from Gao et al. 2024,
    "Scaling and evaluating sparse autoencoders": encoder initialised as the
    transpose of a unit-norm decoder, decoder columns kept at unit norm, an
    auxiliary TopK loss (AuxK) that trains dead latents on the main model's
    residual, a warmup + cosine learning-rate schedule, a run-specific seed and
    globally shuffled batches.

``Recipe.legacy()``
    The recipe the first campaign used (sae-borzoi defaults, lr 1e-5 constant,
    6000 steps, independent default init, free decoder norm, shard-local
    shuffling and the raw seed). Kept so old checkpoints can be reproduced and
    so a pilot can compare the two on equal data.

Both write the same inference checkpoint, so everything downstream is
unchanged.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ag_sae.vendor.sae_borzoi import LN, SparseAutoencoder

RECIPE_VERSION = 2


@dataclass(frozen=True)
class Recipe:
    """Training settings.

    expansion, topk_pct and the MSE reconstruction loss are the published
    sae-borzoi values. Everything else is ours to choose and justify: the
    reference publishes the model, not its training loop.
    """

    expansion: int = 4
    topk_pct: float = 0.05
    #: "dictionary" applies the percentage to the expanded latents (the paper);
    #: "input" applies it to the input channels (the released script).
    topk_basis: str = "dictionary"
    #: Explicit k. Overrides topk_pct when positive (the k=64 study uses 64).
    topk: int = 0
    lr: float = 2e-4
    #: Optimiser steps, not epochs. Epoch count depends on pool size, so it is
    #: the wrong unit once the pool changes.
    steps: int = 15000
    batch_tokens: int = 16384
    #: Rows per forward/backward. With the data on the GPU there is no reason
    #: to accumulate; lower it only if memory runs out.
    microbatch: int = 16384
    checkpoint_every: int = 500
    eval_every: int = 500
    seed: int = 0
    #: Fixed dictionary width. Set this to compare taps of different widths on
    #: equal terms; 0 means derive it from `expansion`.
    n_features: int = 0

    # --- learning-rate schedule -------------------------------------------
    #: "constant" or "cosine" (decays to lr * lr_final_ratio after warmup).
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 500
    lr_final_ratio: float = 0.1

    # --- dead-latent defences -----------------------------------------------
    #: "tied": unit-norm random decoder, encoder = decoder.T (Gao et al.).
    #: "default": independent nn.Linear init (the published recipe).
    init: str = "tied"
    #: "unit": decoder columns renormalised after every step, gradient
    #: component along each column removed first. "free": no constraint.
    decoder_norm: str = "unit"
    #: "mean": pre_bias starts at the mean token-normalised training input.
    #: "zero": the published zero init.
    pre_bias_init: str = "mean"
    #: Auxiliary TopK loss on dead latents. 0 disables it.
    aux_k: int = 512
    aux_coef: float = 1 / 32
    #: A latent that has not fired for this many training tokens is dead.
    dead_tokens: int = 10_000_000

    # --- independence between runs -----------------------------------------
    #: "run": init and data-order seeds are derived from (seed, tap, data), so
    #: two taps or folds trained with --seed 0 do not share an initialisation.
    #: "global": torch.manual_seed(seed) for every run (the first campaign).
    seed_scope: str = "run"
    #: "global": every batch is drawn from the whole split. "shard": shards in
    #: random order, rows shuffled within one shard (the first campaign; each
    #: step then sees ~8 Mb of a single chromosome).
    shuffle: str = "global"

    @classmethod
    def legacy(cls, **overrides) -> "Recipe":
        """The recipe of the first campaign, for reproduction and pilots."""
        base = dict(lr=1e-5, steps=6000, microbatch=1024, checkpoint_every=100,
                    eval_every=250, lr_schedule="constant", lr_warmup_steps=0,
                    init="default", decoder_norm="free", pre_bias_init="zero",
                    aux_k=0, aux_coef=0.0, seed_scope="global", shuffle="shard")
        base.update(overrides)
        return cls(**base)

    def hidden(self, d_in: int) -> int:
        return self.n_features or self.expansion * d_in

    def k(self, d_in: int) -> int:
        if self.topk > 0:
            return self.topk
        basis = d_in if self.topk_basis == "input" else self.hidden(d_in)
        return int(self.topk_pct * basis)

    @property
    def uses_aux(self) -> bool:
        return self.aux_k > 0 and self.aux_coef > 0

    def validate(self, d_in: int) -> None:
        for name in ("expansion", "steps", "batch_tokens", "microbatch",
                     "checkpoint_every", "eval_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        choices = {"topk_basis": ("input", "dictionary"),
                   "lr_schedule": ("constant", "cosine"),
                   "init": ("tied", "default"), "decoder_norm": ("unit", "free"),
                   "pre_bias_init": ("mean", "zero"),
                   "seed_scope": ("run", "global"), "shuffle": ("global", "shard")}
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if not 0 < self.topk_pct <= 1 or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("invalid sparsity or learning rate")
        if self.microbatch > self.batch_tokens:
            raise ValueError("microbatch cannot exceed batch_tokens")
        if self.batch_tokens % self.microbatch:
            raise ValueError("batch_tokens must be a multiple of microbatch")
        if self.n_features < 0 or self.topk < 0 or self.aux_k < 0:
            raise ValueError("n_features, topk and aux_k cannot be negative")
        if not 1 <= self.k(d_in) <= self.hidden(d_in):
            raise ValueError(f"k={self.k(d_in)} outside 1..{self.hidden(d_in)}")
        if self.lr_warmup_steps < 0 or not 0 <= self.lr_final_ratio <= 1:
            raise ValueError("invalid learning-rate schedule")
        if self.aux_coef < 0 or self.dead_tokens < 1:
            raise ValueError("invalid AuxK settings")

    def resolved(self, d_in: int) -> dict[str, Any]:
        return {**asdict(self), "recipe_version": RECIPE_VERSION, "d_in": d_in,
                "hidden": self.hidden(d_in), "k": self.k(d_in),
                "token_layernorm": True, "optimizer": "Adam",
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
                "loss": "MSE+AuxK" if self.uses_aux else "MSE"}

    def learning_rate(self, step: int) -> float:
        """Learning rate for optimiser step `step` (0-based)."""
        if self.lr_warmup_steps and step < self.lr_warmup_steps:
            return self.lr * (step + 1) / self.lr_warmup_steps
        if self.lr_schedule == "constant":
            return self.lr
        span = max(1, self.steps - self.lr_warmup_steps)
        progress = min(1.0, (step - self.lr_warmup_steps) / span)
        floor = self.lr * self.lr_final_ratio
        return floor + (self.lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def derive_seed(recipe: Recipe, tap: str, data_key: str, purpose: str) -> int:
    """Seed for one purpose ("init" or "data") of one run.

    With seed_scope="global" this is the raw seed, as in the first campaign.
    Otherwise the tap name and the data identity are mixed in, so runs that
    share a --seed but not a tap or a fold get unrelated initialisations and
    batch orders. crc32 rather than hash(): Python salts str hashes per process.
    """
    if recipe.seed_scope == "global":
        return recipe.seed
    words = [recipe.seed, zlib.crc32(tap.encode()), zlib.crc32(data_key.encode()),
             zlib.crc32(purpose.encode())]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


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
        if torch.is_tensor(batch):
            # Reduce where the batch lives; max is exact in any float width.
            if not torch.isfinite(batch).all():
                raise ValueError("non-finite activations while computing channel scale")
            batch_max = batch.amax(0).cpu().double().numpy()
            batch_abs = batch.abs().amax(0).cpu().double().numpy()
        else:
            array = np.asarray(batch, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("non-finite activations while computing channel scale")
            batch_max, batch_abs = array.max(0), np.abs(array).max(0)
        highest = np.maximum(highest, batch_max)
        largest_abs = np.maximum(largest_abs, batch_abs)
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

    @property
    def n_feat(self) -> int:
        """Dictionary width used by the sparse matching interface."""
        return self.hidden

    def forward(self, raw: torch.Tensor):
        """Return (reconstruction, sparse codes, scaled input).

        The loss is taken in scaled space, as in the upstream training loop.
        """
        scaled = raw / self.channel_scale
        pre, params = self.core.encode(scaled)
        codes = self.core.get_sparse_activations(self.core.activation(pre))
        return self.core.decode(codes, params), codes, scaled

    def forward_train(self, raw: torch.Tensor, dead: torch.Tensor | None = None,
                      aux_k: int = 0):
        """`forward`, plus the AuxK reconstruction from dead latents.

        Returns (reconstruction, codes, scaled input, aux). aux is None when
        nothing is dead. Otherwise it is the decoder applied to the top-`aux_k`
        dead latents only, without pre_bias or the token mean, because its
        target is the main model's residual ``scaled - reconstruction``, which
        has neither. Same scaled space as the main loss.
        """
        scaled = raw / self.channel_scale
        pre, params = self.core.encode(scaled)
        active = self.core.activation(pre)
        codes = self.core.get_sparse_activations(active)
        recon = self.core.decode(codes, params)
        aux = None
        if dead is not None and aux_k > 0:
            n_dead = int(dead.sum())
            if n_dead:
                candidates = active.masked_fill(~dead, 0.0)
                top = torch.topk(candidates, k=min(aux_k, n_dead), dim=-1)
                aux_codes = torch.zeros_like(active).scatter_(-1, top.indices, top.values)
                aux = self.core.decoder(aux_codes) * params["std"]
        return recon, codes, scaled, aux

    @torch.no_grad()
    def encode_topk(self, raw: torch.Tensor):
        """Return positive TopK values and indices without densifying codes."""
        if raw.ndim != 2 or raw.shape[1] != self.d_in:
            raise ValueError(f"Expected (n, {self.d_in}) input, got {tuple(raw.shape)}")
        scaled = raw / self.channel_scale
        pre, _ = self.core.encode(scaled)
        return torch.topk(self.core.activation(pre), k=self.k, dim=-1)

    @torch.no_grad()
    def reconstruct(self, raw: torch.Tensor):
        """Reconstruction back in raw units, plus the codes."""
        recon, codes, _ = self(raw)
        return recon * self.channel_scale, codes

    # --- edits used by the intervention engines --------------------------

    @torch.no_grad()
    def encode_codes(self, raw: torch.Tensor):
        """Dense sparse codes and the token statistics needed to decode them."""
        pre, params = self.core.encode(raw.float() / self.channel_scale)
        return self.core.get_sparse_activations(self.core.activation(pre)), params

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor, params) -> torch.Tensor:
        """Codes back to raw tap units."""
        return self.core.decode(codes, params) * self.channel_scale

    # --- training-time constraints ----------------------------------------

    @torch.no_grad()
    def init_tied_(self) -> None:
        """Unit-norm random decoder columns, encoder = decoder transpose."""
        decoder = self.core.decoder.weight            # (d_in, hidden)
        decoder.normal_()
        decoder.div_(decoder.norm(dim=0, keepdim=True).clamp_min(1e-12))
        self.core.encoder.weight.copy_(decoder.T)

    @torch.no_grad()
    def init_pre_bias_(self, raw_sample: torch.Tensor) -> None:
        """Start pre_bias at the mean token-normalised input of a sample."""
        normed, _, _ = LN(raw_sample.float().to(self.channel_scale.device)
                          / self.channel_scale)
        self.core.pre_bias.copy_(normed.mean(0))

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        decoder = self.core.decoder.weight
        decoder.div_(decoder.norm(dim=0, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def project_decoder_grad_(self) -> None:
        """Remove the gradient component that would only change column norms.

        Renormalisation undoes the radial part of each update anyway; leaving
        it in the gradient would let Adam's per-element scaling turn some of
        it into a change of direction.
        """
        weight, grad = self.core.decoder.weight, self.core.decoder.weight.grad
        if grad is not None:
            grad.sub_((grad * weight).sum(0, keepdim=True) * weight)

    def inference_state(self) -> dict[str, torch.Tensor]:
        """The tensors `concepts.FrozenSAE` expects, with flat key names."""
        return {"encoder.weight": self.core.encoder.weight.detach().cpu().clone(),
                "decoder.weight": self.core.decoder.weight.detach().cpu().clone(),
                "latent_bias": self.core.latent_bias.detach().cpu().clone(),
                "pre_bias": self.core.pre_bias.detach().cpu().clone(),
                "channel_scale": self.channel_scale.detach().cpu().clone()}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "BorzoiSAE":
        """Load the inference checkpoint written by ``save_inference_checkpoint``."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = blob.get("state", {})
        recipe = blob.get("recipe", {})
        required = {"encoder.weight", "decoder.weight", "latent_bias", "pre_bias",
                    "channel_scale"}
        missing = sorted(required - set(state))
        if missing:
            raise ValueError(f"Inference checkpoint is missing tensors: {missing}")
        if not all(torch.is_tensor(state[name]) and torch.isfinite(state[name]).all()
                   for name in required):
            raise ValueError("Inference checkpoint contains invalid tensors")
        encoder = state["encoder.weight"]
        hidden, d_in = encoder.shape
        k = int(recipe.get("k", 0))
        if recipe.get("d_in", d_in) != d_in or recipe.get("hidden", hidden) != hidden:
            raise ValueError("Checkpoint recipe disagrees with tensor dimensions")
        model = cls(d_in, hidden, k, state["channel_scale"])
        with torch.no_grad():
            model.core.encoder.weight.copy_(encoder)
            model.core.decoder.weight.copy_(state["decoder.weight"])
            model.core.latent_bias.copy_(state["latent_bias"])
            model.core.pre_bias.copy_(state["pre_bias"])
        return model.to(device).eval().requires_grad_(False)


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


def build(d_in: int, recipe: Recipe, scale, device: str = "cpu", *,
          seed: int | None = None) -> BorzoiSAE:
    """Create a model on `device` with the recipe's dimensions and weight init.

    `seed` defaults to recipe.seed (the first campaign's behaviour); `train`
    passes the run-specific seed from `derive_seed`. The data-dependent
    pre_bias init needs training rows and is done by `train`.
    """
    recipe.validate(d_in)
    torch.manual_seed(recipe.seed if seed is None else seed)
    model = BorzoiSAE(d_in, recipe.hidden(d_in), recipe.k(d_in), scale)
    if recipe.init == "tied":
        model.init_tied_()
    elif recipe.decoder_norm == "unit":
        model.normalize_decoder_()
    return model.to(device)
