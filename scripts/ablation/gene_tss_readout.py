"""Gene and TSS readouts for one-bin full-stream SAE injection.

RNA coverage is averaged over the selected MANE transcript's exon bases, one
strand-matched total RNA-seq track at a time. CAGE and PRO-cap are summed in
five 128-bp bins centered on the annotated TSS. Track identities are frozen in
the plan; no channel average is used to choose a favorable tissue afterward.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def match_control_norm(
    baseline: torch.Tensor, target: torch.Tensor, control: torch.Tensor,
    bin_index: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match the control's tap-space L2 change to the target at the same bin.

    This uses only latent activations, before any output is examined. Scaling
    stays on the control decoder direction because the SAE decoder is linear in
    its codes with the original token's normalization parameters held fixed.
    """
    if (baseline.shape != target.shape or baseline.shape != control.shape
            or baseline.ndim != 3 or baseline.shape[0] != 1
            or not 0 <= bin_index < baseline.shape[1]):
        raise ValueError("Mismatched tower activations or TSS bin index")
    target_delta = (target[0, bin_index] - baseline[0, bin_index]).float()
    control_delta = (control[0, bin_index] - baseline[0, bin_index]).float()
    target_norm = float(torch.linalg.vector_norm(target_delta))
    control_norm = float(torch.linalg.vector_norm(control_delta))
    if not np.isfinite(target_norm) or not np.isfinite(control_norm) or min(
        target_norm, control_norm
    ) <= 1e-8:
        raise ValueError("Target or control has no effective injection at the TSS")
    ratio = target_norm / control_norm
    matched = control.clone()
    matched[0, bin_index] = (baseline[0, bin_index].float()
                             + control_delta * ratio).to(control.dtype)
    achieved_norm = float(torch.linalg.vector_norm(
        (matched[0, bin_index] - baseline[0, bin_index]).float()
    ))
    if not np.isclose(achieved_norm, target_norm, rtol=1e-4, atol=1e-5):
        raise ValueError("Control intervention could not be dose matched")
    return matched, {"target_l2": target_norm, "control_before_l2": control_norm,
                     "control_scale": ratio, "control_matched_l2": achieved_norm}


def extract_gene_readouts(output: dict, gene: dict, tracks: dict) -> dict[str, np.ndarray]:
    """Extract one scalar per predeclared track from a model prediction."""
    exon_bins = np.array([item["bin_index"] for item in gene["exon_bin_weights"]], dtype=np.int64)
    exon_bp = np.array([item["exon_bp"] for item in gene["exon_bin_weights"]], dtype=np.float64)
    tss_bin = int(gene["tss_bin_index"])
    if (not len(exon_bins) or not 0 <= tss_bin < 8192
            or (exon_bins < 0).any() or (exon_bins >= 8192).any()
            or (exon_bp <= 0).any() or (exon_bp > 128).any()):
        raise ValueError("Gene exon or TSS bin falls outside the 1-Mb readout")
    result = {}
    for head in ("rna_seq", "cage", "procap"):
        data = output[head]
        if not isinstance(data, dict) or 128 not in data:
            raise ValueError(f"Missing 128-bp {head} readout")
        tensor = data[128]
        if tensor.ndim != 3 or tensor.shape[:2] != (1, 8192):
            raise ValueError(f"Unexpected {head} readout shape")
        indices = [int(track["index"]) for track in tracks[head]]
        if (not indices or len(indices) != len(set(indices)) or min(indices) < 0
                or max(indices) >= tensor.shape[2]):
            raise ValueError(f"Invalid frozen {head} track indices")
        if head == "rna_seq":
            values = tensor[0, exon_bins][:, indices].float()
            weights = torch.as_tensor(exon_bp / exon_bp.sum(), device=values.device,
                                      dtype=values.dtype)
            per_track = (values * weights[:, None]).sum(dim=0)
        else:
            left, right = max(0, tss_bin - 2), min(8192, tss_bin + 3)
            per_track = tensor[0, left:right, indices].float().sum(dim=0)
        if not bool(torch.isfinite(per_track).all()) or bool((per_track < 0).any()):
            raise ValueError(f"Nonfinite or negative {head} predicted signal")
        result[head] = per_track.cpu().numpy()
    return result


def summarize_gene_effects(readouts: dict[str, np.ndarray], tracks: dict) -> dict:
    """Describe paired target/control effects without treating tracks as replicas."""
    summary = {}
    for head in ("rna_seq", "cage", "procap"):
        original = readouts[f"{head}_original"]
        baseline = readouts[f"{head}_reconstruction"]
        target = readouts[f"{head}_target"]
        control = readouts[f"{head}_control"]
        n_tracks = len(tracks[head])
        if any(value.shape != (n_tracks,) or not np.isfinite(value).all()
               or (value < 0).any() for value in
               (original, baseline, target, control)):
            raise ValueError(f"Invalid per-track {head} values")
        offset = 0.001 if head == "rna_seq" else 1.0
        target_lfc = np.log2((target + offset) / (baseline + offset))
        control_lfc = np.log2((control + offset) / (baseline + offset))
        recon_lfc = np.log2((baseline + offset) / (original + offset))
        readouts[f"{head}_target_log2_ratio"] = target_lfc
        readouts[f"{head}_control_log2_ratio"] = control_lfc
        readouts[f"{head}_reconstruction_log2_ratio"] = recon_lfc
        summary[head] = {
            "n_tracks": n_tracks,
            "target_median_log2_ratio": float(np.median(target_lfc)),
            "control_median_log2_ratio": float(np.median(control_lfc)),
            "paired_target_minus_control_median": float(np.median(target_lfc - control_lfc)),
            "fraction_target_increase": float(np.mean(target_lfc > 0)),
            "reconstruction_median_abs_log2_ratio": float(np.median(np.abs(recon_lfc))),
        }
        if head == "rna_seq":
            # Baseline-only track selection: no target/control difference can
            # influence which tissue is reported as the primary RNA readout.
            # Track lists are frozen in index order, so argmax has a stable tie.
            primary = int(np.argmax(original))
            track = tracks[head][primary]
            summary[head]["highest_original_expression_track"] = {
                "list_position": primary,
                "track_index": int(track["index"]),
                "biosample": track.get("biosample"),
                "original": float(original[primary]),
                "reconstruction": float(baseline[primary]),
                "target": float(target[primary]),
                "control": float(control[primary]),
                "target_log2_ratio": float(target_lfc[primary]),
                "control_log2_ratio": float(control_lfc[primary]),
                "target_minus_control_log2_ratio": float(
                    target_lfc[primary] - control_lfc[primary]
                ),
                "reconstruction_log2_ratio": float(recon_lfc[primary]),
            }
    return summary
