"""Turn a feature off inside the model and measure what the model loses.

Concept matching says a feature tracks a concept. That is a correlation. This
asks the causal question: if the feature's contribution is removed from the
activation the model goes on to use, do the model's own predictions change, and
do they change more at the concept's sites than elsewhere?

The ablation is done in the SAE's basis. For the Borzoi SAE, interventions must
invert both per-channel scaling and the row-wise LayerNorm used by the encoder.
The low-level matrix helper below is only valid when activations and decoder
directions are already expressed in the same coordinate system; use
``ablate_sae_features`` for raw activations and a BorzoiSAE checkpoint.

Two controls decide whether a result means anything.

* A matched random feature. Removing any direction degrades a model a little,
  so without this every ablation looks causal.
* Sites away from the concept. A feature that matters everywhere is not
  evidence about the concept, only about the model's sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def decoder_directions(checkpoint: str) -> np.ndarray:
    """(n_features, d_in) decoder columns in the SAE's normalized space.

    These are not raw AlphaGenome-space directions: converting them to a raw
    activation edit also needs the row's LayerNorm std and channel_scale.
    """
    import torch

    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weight = blob["state"]["decoder.weight"].detach().cpu().float().numpy()
    # torch Linear stores (out, in) = (d_in, n_features).
    return np.ascontiguousarray(weight.T)


def ablate_activation(
    activation: np.ndarray,
    codes: np.ndarray,
    directions: np.ndarray,
    features: Sequence[int],
) -> np.ndarray:
    """Subtract contributions when all inputs are already in the same space.

    This is a low-level linear helper. It does not invert BorzoiSAE's
    per-channel scaling or row-wise LayerNorm. For raw model activations, use
    :func:`ablate_sae_features` instead.
    """
    activation = np.asarray(activation, dtype=np.float32)
    codes = np.asarray(codes, dtype=np.float32)
    if codes.shape[0] != activation.shape[0]:
        raise ValueError("codes and activation must have the same number of rows")
    if directions.shape[1] != activation.shape[1]:
        raise ValueError("directions must match the activation width")
    features = np.asarray(list(features), dtype=np.int64)
    if features.size == 0:
        return activation.copy()
    if (features < 0).any() or (features >= directions.shape[0]).any():
        raise ValueError("feature index outside the dictionary")
    return activation - codes[:, features] @ directions[features]


def ablate_sae_features(sae, activation, features: Sequence[int],
                        positions: Sequence[int] | None = None):
    """Remove SAE latents from raw activation rows, preserving the SAE residual.

    ``activation`` must be a torch tensor with shape ``(positions, channels)``.
    The decoder difference is reconstructed with the exact row-wise LayerNorm
    parameters and channel scaling from the checkpoint, then added to the raw
    activation. If ``positions`` is supplied, all other rows remain untouched.
    """
    import torch

    if not torch.is_tensor(activation) or activation.ndim != 2:
        raise ValueError("activation must be a 2-D torch tensor")
    if activation.shape[1] != sae.d_in:
        raise ValueError("activation width does not match the SAE")
    chosen = np.asarray(list(features), dtype=np.int64)
    if positions is None:
        rows = np.arange(len(activation), dtype=np.int64)
    else:
        rows = np.asarray(list(positions), dtype=np.int64)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= len(activation)).any():
            raise ValueError("position index outside activation rows")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("positions must not contain duplicates")
    if chosen.size == 0 or rows.size == 0:
        return activation.clone()
    if (chosen < 0).any() or (chosen >= sae.hidden).any():
        raise ValueError("feature index outside the dictionary")
    if len(np.unique(chosen)) != len(chosen):
        raise ValueError("features must not contain duplicates")

    device = sae.channel_scale.device
    if activation.device != device:
        raise ValueError("SAE and activation must be on the same device")
    scale = sae.channel_scale.float()
    raw = activation[torch.as_tensor(rows, dtype=torch.long, device=device)].float()
    hidden, params = sae.core.encode(raw / scale)
    codes = sae.core.get_sparse_activations(sae.core.activation(hidden))
    edited_codes = codes.clone()
    edited_codes[:, chosen.tolist()] = 0
    reconstructed = sae.core.decode(codes, params) * scale
    changed = sae.core.decode(edited_codes, params) * scale
    if not torch.isfinite(reconstructed).all() or not torch.isfinite(changed).all():
        raise ValueError("Nonfinite SAE reconstruction or feature edit")

    result = activation.clone()
    row_tensor = torch.as_tensor(rows, dtype=torch.long, device=device)
    result[row_tensor] = (raw + changed - reconstructed).to(activation.dtype)
    return result


def matched_controls(
    activity: np.ndarray,
    features: Sequence[int],
    n_controls: int,
    rng: np.random.Generator,
    tolerance: float = 0.25,
) -> list[int]:
    """Pick features that fire about as often and as hard as the targets.

    Without matching, a control drawn uniformly is usually a rarely firing
    feature, removing it changes nothing, and the target looks causal by
    comparison. `activity` is one summary per feature, e.g. mean activation.
    """
    activity = np.asarray(activity, dtype=float)
    targets = set(int(f) for f in features)
    if not targets:
        raise ValueError("No target features")
    pool = np.array([i for i in range(activity.size) if i not in targets])
    if pool.size == 0:
        raise ValueError("No features left to draw controls from")

    chosen: list[int] = []
    for target in sorted(targets):
        want = activity[target]
        window = np.abs(activity[pool] - want) <= tolerance * max(abs(want), 1e-12)
        candidates = pool[window] if window.any() else pool
        take = min(n_controls, candidates.size)
        chosen.extend(int(c) for c in rng.choice(candidates, take, replace=False))
    return chosen


@dataclass(frozen=True)
class AblationResult:
    per_site: pd.DataFrame
    summary: dict


def run_ablation(
    predict: Callable[[np.ndarray], np.ndarray],
    activation: np.ndarray,
    codes: np.ndarray,
    directions: np.ndarray,
    labels: np.ndarray,
    features: Sequence[int],
    *,
    control_features: Sequence[int] | None = None,
    name: str = "",
) -> AblationResult:
    """Measure the per-site effect of removing `features`, against controls.

    `predict` maps an activation matrix to the model's output for those rows.
    It is called three times at most: once intact, once ablated, once for the
    control set. The effect is the absolute change per site, summed over
    whatever the output carries.
    """
    baseline = np.asarray(predict(activation), dtype=np.float64)
    if baseline.shape[0] != activation.shape[0]:
        raise ValueError("predict must return one row per site")

    def effect(chosen: Sequence[int]) -> np.ndarray:
        changed = predict(ablate_activation(activation, codes, directions, chosen))
        delta = np.asarray(changed, dtype=np.float64) - baseline
        return np.abs(delta).reshape(len(delta), -1).sum(axis=1)

    labels = np.asarray(labels, dtype=bool)
    if labels.size != activation.shape[0]:
        raise ValueError("labels must describe the same sites")

    per_site = pd.DataFrame({
        "is_concept": labels,
        "effect": effect(features),
    })
    if control_features is not None:
        per_site["control_effect"] = effect(control_features)

    on = per_site.effect[labels]
    off = per_site.effect[~labels]
    if on.empty or off.empty:
        raise ValueError("Need sites both inside and outside the concept")
    summary = {
        "name": name,
        "features": [int(f) for f in features],
        "n_sites": int(len(per_site)),
        "n_concept_sites": int(labels.sum()),
        "effect_at_concept": float(on.mean()),
        "effect_elsewhere": float(off.mean()),
        # The number to read. A feature that matters everywhere has a ratio of
        # one however large its raw effect is.
        "selectivity": float(on.mean() / off.mean()) if off.mean() > 0 else float("inf"),
    }
    if control_features is not None:
        control_on = per_site.control_effect[labels]
        control_off = per_site.control_effect[~labels]
        summary["control_features"] = [int(f) for f in control_features]
        summary["control_effect_at_concept"] = float(control_on.mean())
        summary["control_selectivity"] = (
            float(control_on.mean() / control_off.mean())
            if control_off.mean() > 0 else float("inf"))
        summary["selectivity_over_control"] = (
            summary["selectivity"] - summary["control_selectivity"])
    return AblationResult(per_site, summary)


def paired_permutation_p(
    effect: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray | None = None,
    n_permutations: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided p for "the effect is larger at the concept's sites".

    Sites inside one window are not independent, so the label pattern is moved
    between whole windows rather than between sites. Each window keeps its
    internal arrangement and only swaps places with another window of the same
    size. Permuting single sites would treat thousands of correlated bins as
    thousands of observations and report a p-value orders of magnitude too
    small.

    With no clusters given, every site is its own cluster, which is the
    independence assumption and is almost always wrong here.
    """
    effect = np.asarray(effect, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if effect.size != labels.size:
        raise ValueError("effect and labels must describe the same sites")
    if not labels.any() or labels.all():
        raise ValueError("Need sites both inside and outside the concept")

    if clusters is None:
        clusters = np.arange(effect.size)
    clusters = np.asarray(clusters)
    if clusters.size != effect.size:
        raise ValueError("clusters must describe the same sites")

    # Rows belonging to each cluster, and the label pattern that came with it.
    members = [np.flatnonzero(clusters == key) for key in np.unique(clusters)]
    patterns = [labels[rows] for rows in members]
    by_size: dict[int, list[int]] = {}
    for index, rows in enumerate(members):
        by_size.setdefault(rows.size, []).append(index)

    observed = effect[labels].mean() - effect[~labels].mean()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_permutations):
        shuffled = np.empty_like(labels)
        for group in by_size.values():
            order = rng.permutation(len(group))
            for slot, source in enumerate(order):
                shuffled[members[group[slot]]] = patterns[group[source]]
        if not shuffled.any() or shuffled.all():
            continue
        if effect[shuffled].mean() - effect[~shuffled].mean() >= observed:
            hits += 1
    return float((1 + hits) / (1 + n_permutations))


# --------------------------------------------------------------------------
# Putting the ablation inside a real forward pass
# --------------------------------------------------------------------------

#: Read from the port's source, not assumed. `encoder` returns
#: `(trunk, intermediates)`; the trunk is the main path into the tower, while
#: the intermediates are U-Net skip connections consumed by
#: `decoder(trunk, intermediates)`, which produces the 1 bp resolution output.
#: So the two kinds of tap do not intervene on the same thing:
#:
#:   tower hook -> MHA branch at that block, then downstream tower and heads
#:   conv tap   -> the skip connection only, so the 1 bp heads and not the
#:                 128 bp ones, and it asks what the skip carries rather than
#:                 what the conv layer computes
#:
#: A conv-tap ablation scored against 128 bp tracks would show no effect and
#: the reason would be architectural, not biological.
TOWER_AFFECTS = "MHA branch and downstream outputs (not pair-update or shortcut input)"
CONV_AFFECTS = "1 bp resolution outputs only"


class Intervention:
    """Edit the selected branch at a tap during the forward pass.

    `replace` receives the captured activation as (positions, channels) and
    returns the same shape. Tower taps hook the MHA input only; they do not
    rewrite the pair-update or residual-shortcut inputs. This class therefore
    implements an attention-path diagnostic, not a full-stream block edit.
    Install once, call the model, remove.
    """

    def __init__(self, model, tap, replace: Callable[[np.ndarray], np.ndarray]) -> None:
        self.model = model
        self.tap = tap
        self.replace = replace
        self._handles: list = []

    def _as_tensor(self, values, like):
        import torch

        return torch.as_tensor(np.ascontiguousarray(values),
                               dtype=like.dtype, device=like.device)

    def __enter__(self) -> "Intervention":
        if self.tap.kind == "tower":
            block = self.model.tower.blocks[int(self.tap.key)]

            def on_mha(_module, args, kwargs):
                # args[0] is the residual stream entering the block, NLC.
                stream = args[0]
                edited = self.replace(stream[0].detach().float().cpu().numpy())
                new = stream.clone()
                new[0] = self._as_tensor(edited, stream)
                return (new,) + tuple(args[1:]), kwargs

            self._handles.append(
                block["mha"].register_forward_pre_hook(on_mha, with_kwargs=True))
        elif self.tap.kind == "encoder":
            def on_encoder(_module, _args, output):
                trunk, intermediates = output
                current = intermediates[self.tap.key]
                edited = self.replace(current[0].detach().float().cpu().numpy().T)
                new = current.clone()
                new[0] = self._as_tensor(edited.T, current)
                intermediates[self.tap.key] = new
                return trunk, intermediates

            self._handles.append(self.model.encoder.register_forward_hook(on_encoder))
        else:
            raise ValueError(f"Unknown tap kind {self.tap.kind!r}")
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def feature_remover(
    codes: np.ndarray, directions: np.ndarray, features: Sequence[int],
    positions: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a low-level linear replacement for activations already in one space.

    `positions` maps each row of `codes` to a row of the activation, for the
    case where only a sample of positions was encoded. Rows the SAE never saw
    are left untouched rather than guessed at. This helper does not undo
    BorzoiSAE's row normalization or channel scaling; use
    :func:`ablate_sae_features` for raw Borzoi activations.
    """
    def replace(activation: np.ndarray) -> np.ndarray:
        out = np.array(activation, dtype=np.float32, copy=True)
        if positions is None:
            return ablate_activation(out, codes, directions, features)
        rows = np.asarray(positions, dtype=np.int64)
        if rows.size != codes.shape[0]:
            raise ValueError("positions must have one entry per code row")
        out[rows] = ablate_activation(out[rows], codes, directions, features)
        return out
    return replace
