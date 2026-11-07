"""AlphaGenome and its SAE features as tangermeme models.

tangermeme (Schreiber, github.com/jmschrei/tangermeme) works with any torch
module that maps one-hot sequences shaped (N, 4, L) to tensors. The wrappers
here give it that interface, so the downstream experiments use its tools
instead of hand-written loops:

* `tangermeme.predict.predict` batches the examples and moves them to the GPU;
* `tangermeme.ersatz` makes the sequence perturbations (motif shuffles,
  substitutions, dinucleotide-shuffled backgrounds);
* `tangermeme.marginalize` / `tangermeme.ablate` run insert-a-motif and
  remove-a-motif experiments against SAE features;
* memelite's FIMO (tangermeme's motif scanner) finds motif sites with proper
  p-values instead of a fraction-of-maximum score cut.

Two models are provided:

`FactorialMotifModel`
    The 2x2 motif mediation design (WT, ISM, SAE ablation, SAE rescue) for one
    SAE feature, as one batched forward per window (`edit.run_conditions`).

`SAEFeatureModel`
    SAE codes of chosen features at every bin of a tap. Runs AlphaGenome only
    up to that tap.

AlphaGenome itself takes (B, L, 4); the wrappers transpose.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from ag_sae.edit import (Condition, capture_residual, head_tensor, run_conditions,
                         sae_edit, set_feature_from, zero_feature)

FACTORIAL_CONDITIONS: tuple[str, ...] = ("wt", "ism", "ablation", "rescue")


def to_tangermeme(onehot: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(L, 4) or (N, L, 4) one-hot to tangermeme's (N, 4, L) float32."""
    x = torch.as_tensor(np.asarray(onehot) if not torch.is_tensor(onehot) else onehot)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError(f"expected (N, L, 4), got {tuple(x.shape)}")
    return x.transpose(1, 2).float().contiguous()


def to_alphagenome(x: torch.Tensor) -> torch.Tensor:
    """tangermeme's (N, 4, L) to AlphaGenome's (N, L, 4)."""
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError(f"expected (N, 4, L), got {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


class FactorialMotifModel(nn.Module):
    """2x2 motif mediation of one SAE feature, as a tangermeme model.

    forward(X_wt, X_mut, anchor) with X_* shaped (B, 4, L) and anchor (B,)
    holding the bin to edit returns

    * readouts (B, 4, n_heads): per condition, each head's mean over all its
      tracks and over bins anchor-radius..anchor+radius;
    * latents (B, 2): the feature's code at the anchor in WT and in ISM.

    Conditions, all with the anchor bin replaced by its SAE reconstruction so
    every contrast is between reconstructions:

    wt        natural sequence
    ism       motif core shuffled
    ablation  natural sequence, feature set to 0
    rescue    shuffled sequence, feature restored to its WT value

    `anchor` arrives as float when called through tangermeme.predict (it casts
    every argument); it is rounded back to an index here.
    """

    def __init__(self, model: Any, sae: Any, block_index: int, feature: int,
                 heads: Sequence[str], *, radius: int = 1, max_batch: int = 4,
                 organism: int = 0) -> None:
        super().__init__()
        self.model, self.sae = model, sae
        self.block_index, self.feature = int(block_index), int(feature)
        self.heads, self.radius = tuple(heads), int(radius)
        self.max_batch, self.organism = int(max_batch), int(organism)
        if not 0 <= self.feature < sae.hidden:
            raise ValueError(f"feature {feature} outside the dictionary")

    def conditions(self, anchor: int) -> list[Condition]:
        bins, f = [anchor], self.feature
        return [Condition("wt", 0, bins, sae_edit(self.sae, "wt")),
                Condition("ism", 1, bins, sae_edit(self.sae, "mut")),
                Condition("ablation", 0, bins, sae_edit(self.sae, "wt_ablation", zero_feature(f))),
                Condition("rescue", 1, bins,
                          sae_edit(self.sae, "mut_rescue", set_feature_from(f, "wt")))]

    def _readout(self, anchor: int):
        def readout(outputs, b):
            values = []
            for head in self.heads:
                tensor = head_tensor(outputs, head)[b]
                lo, hi = max(0, anchor - self.radius), min(tensor.shape[0], anchor + self.radius + 1)
                values.append(tensor[lo:hi].float().mean())
            return torch.stack(values)
        return readout

    @torch.inference_mode()
    def forward(self, X_wt: torch.Tensor, X_mut: torch.Tensor, anchor: torch.Tensor):
        readouts, latents = [], []
        for i in range(X_wt.shape[0]):
            a = int(torch.round(anchor[i]).item())
            sequences = to_alphagenome(torch.stack([X_wt[i], X_mut[i]]))
            memo: dict = {}
            result = run_conditions(self.model, self.block_index, sequences,
                                    self.conditions(a), self._readout(a),
                                    heads=self.heads, organism=self.organism,
                                    max_batch=self.max_batch, memo=memo)
            readouts.append(torch.stack([result[name] for name in FACTORIAL_CONDITIONS]))
            latents.append(torch.stack([memo["wt"][0, self.feature],
                                        memo["mut"][0, self.feature]]).float())
        return torch.stack(readouts), torch.stack(latents)


class SAEFeatureModel(nn.Module):
    """SAE codes of `features` at every bin of the tap: (B, bins, n_features).

    Use it with tangermeme.marginalize (does inserting a motif into neutral
    background switch the feature on?) or tangermeme.ablate (does shuffling a
    motif switch it off?). Only the part of AlphaGenome before the tap runs.
    """

    def __init__(self, model: Any, sae: Any, block_index: int,
                 features: Sequence[int], organism: int = 0) -> None:
        super().__init__()
        self.model, self.sae = model, sae
        self.block_index, self.organism = int(block_index), int(organism)
        self.register_buffer("features", torch.as_tensor(list(features), dtype=torch.long),
                             persistent=False)

    @torch.inference_mode()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        stream = capture_residual(self.model, self.block_index, to_alphagenome(X),
                                  self.organism)
        out = []
        for rows in stream:                              # one sequence at a time
            codes, _ = self.sae.encode_codes(rows.float())
            out.append(codes.index_select(1, self.features.to(codes.device)))
        return torch.stack(out)


# --------------------------------------------------------------------------
# Motif sites and perturbations
# --------------------------------------------------------------------------

def _numba_threading() -> None:
    """Use numba's own thread pool unless the user chose one.

    With torch loaded, numba's OpenMP layer can abort the process (two OpenMP
    runtimes in one process; seen on macOS). The workqueue layer has no such
    dependency and is plenty for scanning a few dozen windows.
    """
    import numba

    if "NUMBA_THREADING_LAYER" not in os.environ:
        numba.config.THREADING_LAYER = "workqueue"

def motif_pwm(motif: Any) -> np.ndarray:
    """(4, width) probability matrix from a JasparMotif or a (width, 4) array."""
    pwm = np.asarray(getattr(motif, "pwm", motif), dtype=np.float64)
    if pwm.ndim != 2 or 4 not in pwm.shape:
        raise ValueError(f"expected a (width, 4) matrix, got {pwm.shape}")
    pwm = pwm if pwm.shape[0] == 4 and pwm.shape[1] != 4 else pwm.T
    return pwm / pwm.sum(0, keepdims=True)


def strongest_sites(X: torch.Tensor | np.ndarray, motif: Any, *, name: str = "motif",
                    threshold: float = 1e-4) -> pd.DataFrame:
    """Best FIMO hit per sequence: columns example, start, end, strand, score, p-value.

    `X` is (N, 4, L). Sequences without a hit at `threshold` are absent.
    """
    _numba_threading()
    from memelite import fimo

    array = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    hits = fimo({name: motif_pwm(motif)}, array.astype(np.float32),
                threshold=threshold, reverse_complement=True)[0]
    if hits.empty:
        return pd.DataFrame(columns=["example", "start", "end", "strand", "score", "p-value"])
    hits = hits.rename(columns={"sequence_name": "example"})
    hits["example"] = hits["example"].astype(int)
    best = hits.sort_values(["p-value", "score"], ascending=[True, False])
    best = best.drop_duplicates("example").sort_values("example")
    return best[["example", "start", "end", "strand", "score", "p-value"]].reset_index(drop=True)


def shuffle_site(X: torch.Tensor, start: int, end: int, seed: int, *,
                 method: str = "shuffle", attempts: int = 10) -> torch.Tensor:
    """Copy of X (1, 4, L) with [start, end) shuffled by tangermeme.ersatz.

    "shuffle" keeps the site's base composition (the first campaign's ISM);
    "dinucleotide" also keeps its dinucleotide counts. Draws again when a
    shuffle happens to reproduce the original site.
    """
    _numba_threading()
    from tangermeme.ersatz import dinucleotide_shuffle, shuffle

    if X.ndim != 3 or X.shape[0] != 1:
        raise ValueError("shuffle_site takes a single sequence shaped (1, 4, L)")
    original = X[..., start:end]
    for attempt in range(attempts):
        if method == "shuffle":
            Y = shuffle(X, start=start, end=end, n=1, random_state=seed + attempt)
        elif method == "dinucleotide":
            Y = dinucleotide_shuffle(X, start=start, end=end, n=1,
                                     random_state=seed + attempt, allow_N=True)
        else:
            raise ValueError("method must be shuffle or dinucleotide")
        Y = Y[:, 0].to(X.dtype)                    # (1, n=1, 4, L) -> (1, 4, L)
        if not torch.equal(Y[..., start:end], original):
            return Y
    raise ValueError(f"could not shuffle site {start}-{end} into a different sequence")


def marginalize_features(model: Any, sae: Any, block_index: int, features: Sequence[int],
                         backgrounds: torch.Tensor, motif: str, *, start: int | None = None,
                         batch_size: int = 1, device: str = "cuda"):
    """tangermeme.marginalize on SAE features: codes before/after inserting `motif`.

    `backgrounds` is (N, 4, L), typically dinucleotide-shuffled windows.
    Returns tangermeme's PerturbationResult with (N, bins, n_features) tensors.
    """
    from tangermeme.marginalize import marginalize

    wrapper = SAEFeatureModel(model, sae, block_index, features)
    return marginalize(wrapper, backgrounds, motif, start=start,
                       batch_size=batch_size, device=device)
