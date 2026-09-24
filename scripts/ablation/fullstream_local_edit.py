"""Residual-preserving SAE feature interventions on explicit 128-bp bins.

At every unselected position the activation is unchanged. At selected
positions, an SAE decoder delta edits only the chosen code contribution while
preserving the original activation's residual from its SAE reconstruction.
"""

from __future__ import annotations

import numpy as np
import torch


def checked_element_mask(labels: np.ndarray, n_positions: int) -> np.ndarray:
    """Require one Boolean value per native tower position and a nonempty mask."""
    mask = np.asarray(labels)
    if mask.ndim != 1 or len(mask) != n_positions or mask.dtype != np.bool_:
        raise ValueError("Element mask must be a Boolean value for every 128-bp position")
    if not mask.any():
        raise ValueError("No element-positive bins: local intervention is undefined")
    return mask.copy()


def paired_ablation_is_measurable(counts: dict[str, int]) -> bool:
    """Do not spend five more model forwards on a silent target/control pair."""
    return (counts["target_changed_on_element"] > 0
            and counts["control_changed_on_element"] > 0
            and counts["target_activation_changed_on_element"] > 0
            and counts["control_activation_changed_on_element"] > 0)


@torch.inference_mode()
def active_feature_support(sae, activation: torch.Tensor,
                          element_mask: np.ndarray, feature: int,
                          batch_size: int = 256) -> np.ndarray:
    """Return element bins where one SAE feature is active in the original tap.

    The support is measured before intervention. It is useful for paired
    target/control ablations: both features can then be edited at exactly the
    target's active element bins, rather than giving a denser control more
    opportunities to perturb the model.
    """
    if (activation.ndim != 3 or activation.shape[0] != 1
            or activation.shape[-1] != sae.d_in or batch_size < 1
            or not 0 <= feature < sae.hidden):
        raise ValueError("Invalid activation, feature, or batch size")
    mask = checked_element_mask(element_mask, activation.shape[1])
    selected = np.flatnonzero(mask)
    active_support = np.zeros_like(mask)
    scale = sae.channel_scale.float().to(activation.device)
    for offset in range(0, len(selected), batch_size):
        positions = torch.as_tensor(
            selected[offset:offset + batch_size], dtype=torch.long,
            device=activation.device,
        )
        raw = activation[0, positions].float()
        pre, _params = sae.core.encode(raw / scale)
        codes = sae.core.get_sparse_activations(sae.core.activation(pre))
        if codes.shape != (len(positions), sae.hidden):
            raise ValueError("Unexpected SAE code shape")
        active_support[positions.cpu().numpy()] = (
            codes[:, feature] > 0
        ).cpu().numpy()
    if not active_support.any():
        raise ValueError("Target feature is silent on all annotated element bins")
    return active_support


@torch.inference_mode()
def local_edit_triplet(
    sae,
    activation: torch.Tensor,
    element_mask: np.ndarray,
    target: int,
    control: int,
    mode: str,
    target_level: float,
    control_level: float,
    batch_size: int = 256,
):
    """Return original baseline, target edit, control edit, and code counts.

    Ablation sets the selected feature's code to zero only at selected bins;
    injection raises that code to at least its dev-calibrated positive p95.
    The edited activation is x + decode(c_edited) - decode(c), so x minus its
    SAE reconstruction is retained exactly. All other codes and downstream
    model operations remain unchanged. Injection here is local gain-of-function
    at selected bins, not absent-site sufficiency.
    """
    if (activation.ndim != 3 or activation.shape[0] != 1
            or activation.shape[-1] != sae.d_in or batch_size < 1
            or mode not in {"ablate", "inject"} or target == control
            or not 0 <= target < sae.hidden or not 0 <= control < sae.hidden):
        raise ValueError("Invalid activation, SAE features, mode, or batch size")
    mask = checked_element_mask(element_mask, activation.shape[1])
    if mode == "inject" and (not np.isfinite(target_level) or target_level <= 0
                              or not np.isfinite(control_level) or control_level <= 0):
        raise ValueError("Injection requires positive finite dev calibration levels")
    if not bool(torch.isfinite(activation).all()):
        raise ValueError("Nonfinite source activation")

    baseline, target_edit, control_edit = (activation.clone() for _ in range(3))
    selected = np.flatnonzero(mask)
    counts = {"element_bins": int(len(selected)), "target_fired_on_element": 0,
              "control_fired_on_element": 0, "target_changed_on_element": 0,
              "control_changed_on_element": 0,
              "target_activation_changed_on_element": 0,
              "control_activation_changed_on_element": 0}
    scale = sae.channel_scale.float().to(activation.device)
    for offset in range(0, len(selected), batch_size):
        positions = torch.as_tensor(
            selected[offset:offset + batch_size], dtype=torch.long,
            device=activation.device,
        )
        raw = activation[0, positions].float()
        pre, params = sae.core.encode(raw / scale)
        codes = sae.core.get_sparse_activations(sae.core.activation(pre))
        if codes.shape != (len(positions), sae.hidden):
            raise ValueError("Unexpected SAE code shape")

        counts["target_fired_on_element"] += int((codes[:, target] > 0).sum().item())
        counts["control_fired_on_element"] += int((codes[:, control] > 0).sum().item())
        target_codes, control_codes = codes.clone(), codes.clone()
        if mode == "ablate":
            target_codes[:, target] = 0
            control_codes[:, control] = 0
        else:
            target_codes[:, target] = torch.clamp_min(codes[:, target], target_level)
            control_codes[:, control] = torch.clamp_min(codes[:, control], control_level)
        counts["target_changed_on_element"] += int(
            (target_codes[:, target] != codes[:, target]).sum().item()
        )
        counts["control_changed_on_element"] += int(
            (control_codes[:, control] != codes[:, control]).sum().item()
        )

        rebuilt = sae.core.decode(codes, params) * scale
        changed_target = sae.core.decode(target_codes, params) * scale
        changed_control = sae.core.decode(control_codes, params) * scale
        if not all(bool(torch.isfinite(value).all()) for value in
                   (rebuilt, changed_target, changed_control)):
            raise ValueError("Nonfinite reconstruction or edited activation")
        rebuilt = rebuilt.to(activation.dtype)
        changed_target = changed_target.to(activation.dtype)
        changed_control = changed_control.to(activation.dtype)
        counts["target_activation_changed_on_element"] += int(
            (changed_target != rebuilt).any(dim=-1).sum().item()
        )
        counts["control_activation_changed_on_element"] += int(
            (changed_control != rebuilt).any(dim=-1).sum().item()
        )
        target_delta = changed_target - rebuilt
        control_delta = changed_control - rebuilt
        target_edit[0, positions] = (raw + target_delta).to(activation.dtype)
        control_edit[0, positions] = (raw + control_delta).to(activation.dtype)
    return baseline, target_edit, control_edit, counts


def summarize_local_effects(
    arrays: dict[str, np.ndarray], heads: tuple[str, ...],
    element_mask: np.ndarray, mode: str,
) -> dict[str, dict[str, float]]:
    """Compare paired readouts, reporting both local and propagated effects."""
    if mode not in {"ablate", "inject"}:
        raise ValueError("Unknown intervention mode")
    mask = checked_element_mask(element_mask, len(element_mask))
    summary = {}
    for head in heads:
        original = arrays[f"{head}_original"]
        baseline = arrays[f"{head}_baseline"]
        target = arrays[f"{head}_{mode}_target"]
        control = arrays[f"{head}_{mode}_control"]
        if any(value.shape != mask.shape or not np.isfinite(value).all()
               for value in (original, baseline, target, control)):
            raise ValueError("Readout shape or finite-value check failed")
        std = float(np.std(baseline) + 1e-9)
        target_delta, control_delta = target - baseline, control - baseline
        record = {
            "target_mean_on_element": float(target_delta[mask].mean()),
            "control_mean_on_element": float(control_delta[mask].mean()),
            "target_minus_control_standardized": float(
                (target_delta[mask].mean() - control_delta[mask].mean()) / std
            ),
            "target_mean_off_element": float(target_delta[~mask].mean()) if (~mask).any() else 0.0,
            "control_mean_off_element": float(control_delta[~mask].mean()) if (~mask).any() else 0.0,
            "baseline_mae_on_element": float(np.abs(baseline[mask] - original[mask]).mean()),
            "baseline_mae_off_element": float(np.abs(baseline[~mask] - original[~mask]).mean())
            if (~mask).any() else 0.0,
            "baseline_spatial_std": std,
        }
        summary[head] = record
    return summary
