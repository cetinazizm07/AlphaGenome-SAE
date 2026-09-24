#!/usr/bin/env python3
"""One held-out PLS window: full-stream, PLS-bin-only SAE feature ablation.

Feature 199 and a matched comparator are zeroed in their own SAE latent code,
at annotated PLS 128-bp bins only. By default the comparator is dev-selected
feature 5861; an outcome-blind per-window activation audit can select another
member of the frozen dev candidate pool. Outside PLS bins the tower input is
unchanged. The common at-site baseline is the SAE reconstruction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from fullstream_local_edit import (
    active_feature_support,
    local_edit_triplet,
    paired_ablation_is_measurable,
    summarize_local_effects,
)
from plan_fullstream_cohort import sha256
from run_cloud_fullstream_control_pilot import FullStreamTower, run_replacement
from run_cloud_ablation_pilot import load_sampled_coordinates
from run_fullstream_local_window import HEADS, load_window_data, read_head_summaries


def parse_overrides(items: list[str]) -> dict[str, Path]:
    overrides = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in overrides:
            raise ValueError(f"Expected unique NAME=PATH override, got {item!r}")
        overrides[name] = Path(value)
    return overrides


def checked_plan(path: Path, window_index: int, parent_override: Path | None,
                 control_override: Path | None,
                 source_overrides: dict[str, Path]):
    """Validate the frozen intervention, control selection, and input bytes."""
    plan = json.loads(path.read_text())
    if (plan.get("format") != "fullstream_pls_local_ablation_plan_v3"
            or plan.get("fold") != "fold1" or plan.get("sae_seed") != 0
            or plan.get("tap") != "resid_pre_b8"
            or plan.get("concept") != "cCRE_PLS"
            or plan.get("target_feature") != 199
            or plan.get("control_feature") != 5861
            or tuple(plan["primary_heads"] + plan["off_target_heads"]) != HEADS
            or not 0 <= window_index < len(plan["windows"])):
        raise ValueError("Unexpected v3 PLS-local ablation plan")
    parent = parent_override or Path(plan["parent_plan_path"])
    control = control_override or Path(plan["control_selection_path"])
    if (sha256(parent) != plan["parent_plan_sha256"]
            or sha256(control) != plan["control_selection_sha256"]):
        raise ValueError("Parent plan or dev-control receipt changed")
    paths = {name: Path(value) for name, value in plan["source_paths"].items()}
    for name, replacement in source_overrides.items():
        if name not in paths:
            raise ValueError(f"Unknown source override {name}")
        paths[name] = replacement
    for name, source in paths.items():
        if sha256(source) != plan["source_sha256"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    window = plan["windows"][window_index]
    if (window["mode"] != "paired_local" or window["positive_bins"] < 1
            or window["end"] - window["start"] != 1_048_576):
        raise ValueError("Window is not a PLS-positive 1-Mb test interval")
    return plan, window, paths, control


def intervention_norms(baseline: torch.Tensor, target: torch.Tensor,
                       comparator: torch.Tensor, mask: np.ndarray) -> dict:
    """Record shared-support raw-tap dose, without output-based selection."""
    positions = torch.as_tensor(np.flatnonzero(mask), device=baseline.device)
    target_l2 = torch.linalg.vector_norm(
        (target - baseline)[0, positions].float(), dim=-1
    ).cpu().numpy()
    comparator_l2 = torch.linalg.vector_norm(
        (comparator - baseline)[0, positions].float(), dim=-1
    ).cpu().numpy()
    return {
        "target_mean_l2_per_intervened_bin": float(target_l2.mean()),
        "comparator_mean_l2_per_intervened_bin": float(comparator_l2.mean()),
        "target_effective_bins": int(np.count_nonzero(target_l2)),
        "comparator_effective_bins": int(np.count_nonzero(comparator_l2)),
        "coactive_bins": int(np.count_nonzero((target_l2 > 0) & (comparator_l2 > 0))),
    }


def validate_control_audit(audit: dict, plan: dict, plan_path: Path,
                           control_path: Path, selection: dict,
                           window_index: int, window: dict,
                           coordinates_sha256: str,
                           activation_sha256: str) -> int:
    """Accept only a frozen-plan, output-blind, same-support dose match."""
    candidates = [int(row["feature"]) for row in selection["closest_local_candidates"]
                 if row["relative_dev_pls_rate_difference"] <= 0.25 + 1e-12
                 and row["relative_strength_difference"] <= 0.25 + 1e-12
                 and row["max_existing_auroc"] < 0.55]
    if (audit.get("format") != "outcome_blind_pls_window_control_audit_v1"
            or "model output" not in audit.get("interpretation", "")
            or audit.get("plan_sha256") != sha256(plan_path)
            or audit.get("selection_sha256") != sha256(control_path)
            or audit.get("window_index") != window_index
            or audit.get("window") != window
            or audit.get("target_feature") != plan["target_feature"]
            or audit.get("candidate_features") != candidates
            or audit.get("coordinates_sha256") != coordinates_sha256
            or audit.get("activation_sha256") != activation_sha256):
        raise ValueError("Control audit does not match this frozen held-out window")
    chosen = audit.get("chosen_feature")
    if chosen is None or int(chosen) not in candidates:
        raise ValueError("No eligible same-window matched control was found")
    row = next((item for item in audit["candidates"]
                if item["feature"] == int(chosen)), None)
    if (row is None or row["target_active_bins"] < 1
            or row["comparator_active_on_target_bins"] != row["target_active_bins"]
            or not 0.75 <= row["comparator_to_target_l2_on_target_bins"] <= 1.25):
        raise ValueError("Selected control fails same-support or dose criteria")
    return int(chosen)


def write_result(out: Path, plan_path: Path, control_path: Path,
                 bins, arrays: dict[str, np.ndarray] | None,
                 receipt: dict, control_audit_path: Path | None = None) -> None:
    """Write a unique measured or explicit unmeasurable result atomically."""
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, out / "plan.json")
    shutil.copy2(control_path, out / "dev_control_selection.json")
    if control_audit_path is not None:
        shutil.copy2(control_audit_path, out / "window_control_audit.json")
    for source in (__file__, Path(__file__).with_name("fullstream_local_edit.py"),
                   Path(__file__).with_name("run_fullstream_local_window.py"),
                   Path(__file__).with_name("run_cloud_fullstream_control_pilot.py")):
        shutil.copy2(source, out / Path(source).name)
    bins[["chrom", "bin_start", "n_mask", "cCRE_PLS"]].to_parquet(
        out / "window_annotations.parquet", index=False
    )
    if arrays is not None:
        temporary = out / "readouts.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(out / "readouts.npz")
        receipt["readouts_sha256"] = sha256(out / "readouts.npz")
    else:
        receipt["readouts_sha256"] = None
    temporary = out / "receipt.json.tmp"
    with temporary.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out / "receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--control-selection", type=Path)
    parser.add_argument(
        "--control-audit", type=Path,
        help="Outcome-blind same-window control assignment from cached activations",
    )
    parser.add_argument("--source-override", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.out.exists() or args.batch_size < 1:
        raise ValueError("Output exists or batch size is invalid")
    plan, window, paths, control = checked_plan(
        args.plan, args.window_index, args.parent_plan,
        args.control_selection, parse_overrides(args.source_override)
    )
    expected, bins, mask, coord_sha, onehot = load_window_data(
        window, paths, args.acts_dir, args.fasta_dir
    )
    control_feature = int(plan["control_feature"])
    if args.control_audit is not None:
        audit = json.loads(args.control_audit.read_text())
        control_selection = json.loads(control.read_text())
        coords, coordinate_path = load_sampled_coordinates(
            args.acts_dir, plan["tap"], window["chrom"], int(window["start"])
        )
        activation_index = json.loads(paths["activation_index"].read_text())
        indexed = [row["taps"][plan["tap"]] for row in activation_index["shards"]
                   if row["split"] == "test" and plan["tap"] in row["taps"]
                   and row["taps"][plan["tap"]]["coordinates"]
                   == str(coordinate_path.relative_to(args.acts_dir))]
        if len(indexed) != 1 or sha256(coordinate_path) != coord_sha:
            raise ValueError("Window activation shard is ambiguous for control audit")
        activation_path = args.acts_dir / indexed[0]["activations"]
        if sha256(activation_path) != indexed[0]["activations_sha256"]:
            raise ValueError("Current held-out activation shard hash changed")
        control_feature = validate_control_audit(
            audit, plan, args.plan, control, control_selection,
            args.window_index, window, coord_sha, sha256(activation_path)
        )
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    tap = TAPS[plan["tap"]]
    if tap.kind != "tower" or tap.bin_bp != 128:
        raise ValueError("Expected full-stream 128-bp tower block input")
    sae = BorzoiSAE.from_checkpoint(paths["sae"], device="cuda")
    if sae.d_in != tap.channels or sae.hidden <= control_feature:
        raise ValueError("Frozen SAE/tap dimensions changed")
    model = AlphaGenome.from_pretrained(
        paths["weights"], dtype_policy=DtypePolicy.mixed_precision(), device="cuda"
    )
    model.eval().requires_grad_(False)
    input_tensor = torch.from_numpy(onehot)[None].to("cuda")
    organism = torch.tensor([0], device="cuda")
    forward_args = {"resolutions": (128,), "heads": HEADS, "channels_last": True}
    arrays: dict[str, np.ndarray] = {"bin_start": expected,
                                     "concept_label": mask}
    captured = []
    started = time.monotonic()

    def capture(value):
        captured.append(np.array(value, dtype=np.float32, copy=True))
        return value

    with torch.inference_mode(), FullStreamTower(model, tap, capture):
        original = model.predict(input_tensor, organism, **forward_args)
    if len(captured) != 1 or captured[0].shape != (8192, tap.channels):
        raise ValueError("Tower capture shape or call count changed")
    read_head_summaries(original, "original", arrays)
    for head in HEADS:
        arrays[f"{head}_baseline"] = arrays[f"{head}_original"].copy()
        arrays[f"{head}_baseline_track_mean"] = arrays[
            f"{head}_original_track_mean"
        ].copy()
    del original
    activation = torch.from_numpy(captured[0])[None].to("cuda")
    del captured
    intervention_mask = active_feature_support(
        sae, activation, mask, plan["target_feature"], args.batch_size
    )
    arrays["intervention_mask"] = intervention_mask.copy()
    baseline, target, comparator, counts = local_edit_triplet(
        sae, activation, intervention_mask, plan["target_feature"],
        control_feature, "ablate", 1.0, 1.0, args.batch_size
    )
    outside = torch.as_tensor(~intervention_mask, device=activation.device)
    if (not torch.equal(baseline[0, outside], activation[0, outside])
            or not torch.equal(target[0, outside], activation[0, outside])
            or not torch.equal(comparator[0, outside], activation[0, outside])):
        raise ValueError("Local intervention changed a tower input outside shared support")
    norms = intervention_norms(baseline, target, comparator, intervention_mask)
    receipt = {
        "format": "fullstream_pls_local_ablation_window_v6",
        "interpretation": (
            "Exploratory residual-preserving full-stream ablation. The raw tower "
            "activation is the baseline; the intervention adds only the decoder "
            "delta for the selected SAE feature at shared target-active PLS bins. "
            "No inferential claim is supported by a single window."
        ),
        "control_selection_caveat": (
            "The candidate pool was filtered using dev-bin activity/tap dose and "
            "an existing max-AUROC threshold computed on test labels; per-window "
            "ranking used cached activations only. This remains exploratory."
        ),
        "selection_plan_interpretation": plan["interpretation"],
        "status": "measured" if paired_ablation_is_measurable(counts)
        else "unmeasurable_local_ablation",
        "plan_sha256": sha256(args.plan),
        "control_selection_sha256": sha256(control),
        "worker_sha256": sha256(Path(__file__)),
        "local_edit_sha256": sha256(Path(__file__).with_name("fullstream_local_edit.py")),
        "fold": "fold1", "sae_seed": 0, "tap": "resid_pre_b8",
        "tap_route": "tower_block_input_before_pair_update_mha_residual_mlp",
        "window_index": args.window_index, "window": window,
        "target_feature": 199, "control_feature": control_feature,
        "control_type": ("per-window same-support and tap-L2 matched member of the "
                         "frozen dev candidate pool; selection used cached activations only"
                         if args.control_audit is not None else
                         "dev-PLS firing-and-tap-L2 matched comparator; not globally firing matched"),
        "window_control_audit_sha256": (
            sha256(args.control_audit) if args.control_audit is not None else None
        ),
        "n_native_positions": 8192,
        "n_concept_positive_bins": int(mask.sum()),
        "n_intervened_bins": int(intervention_mask.sum()),
        "intervention_scope": (
            "same target-active subset of cCRE_PLS-positive 128-bp bins for target and control"
            if args.control_audit is not None else
            "target-active subset of cCRE_PLS-positive 128-bp bins; comparator must be measurable"
        ),
        "baseline_scope": (
            "unmodified tower activation; preserve the SAE residual and add only "
            "the selected feature's decoder delta at shared support"
        ),
        "code_counts": counts, "tap_intervention": norms,
        "source_sha256": plan["source_sha256"],
        "sampled_coordinates_sha256": coord_sha,
        "heads_primary": plan["primary_heads"],
        "heads_off_target": plan["off_target_heads"],
    }
    if receipt["status"] != "measured":
        receipt["elapsed_seconds"] = time.monotonic() - started
        write_result(args.out, args.plan, control, bins, None, receipt,
                     args.control_audit)
        print(json.dumps({"out": str(args.out), "status": receipt["status"],
                          "code_counts": counts,
                          "receipt_sha256": sha256(args.out / "receipt.json")}))
        return
    for name, replacement in (("ablate_target", target),
                              ("ablate_control", comparator)):
        with torch.inference_mode(), FullStreamTower(
            model, tap, run_replacement(replacement)
        ):
            prediction = model.predict(input_tensor, organism, **forward_args)
        read_head_summaries(prediction, name, arrays)
        del prediction
        torch.cuda.empty_cache()
    receipt["effects_on_all_pls_bins"] = summarize_local_effects(
        arrays, HEADS, mask, "ablate"
    )
    receipt["effects_on_target_active_support"] = summarize_local_effects(
        arrays, HEADS, intervention_mask, "ablate"
    )
    receipt["elapsed_seconds"] = time.monotonic() - started
    write_result(args.out, args.plan, control, bins, arrays, receipt,
                 args.control_audit)
    print(json.dumps({"out": str(args.out), "status": receipt["status"],
                      "code_counts": counts, "tap_intervention": norms,
                      "receipt_sha256": sha256(args.out / "receipt.json"),
                      "readouts_sha256": receipt["readouts_sha256"]}))


if __name__ == "__main__":
    main()
