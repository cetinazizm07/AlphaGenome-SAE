#!/usr/bin/env python3
"""One held-out PLS window with paired, element-local SAE edits.

Both ablation and injection edit only the annotated PLS-positive 128-bp bins.
Outside those bins the tower receives its original activation. The injection
arm is local gain-of-function, not the older absent-site sufficiency test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from fullstream_local_edit import (
    checked_element_mask,
    local_edit_triplet,
    paired_ablation_is_measurable,
    summarize_local_effects,
)
from plan_fullstream_cohort import sha256
from run_cloud_ablation_pilot import chrom_sequence, load_sampled_coordinates
from run_cloud_fullstream_control_pilot import FullStreamTower, run_replacement


HEADS = ("dnase", "atac", "cage", "rna_seq", "procap", "chip_tf", "chip_histone")


def load_frozen_plan(
    path: Path, window_index: int,
    parent_override: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
) -> tuple[dict, dict, dict[str, Path]]:
    """Check plan and hashes; allow relocating identical files to another VM."""
    plan = json.loads(path.read_text())
    if (plan.get("format") != "fullstream_pls_local_paired_plan_v2"
            or plan.get("fold") != "fold1" or plan.get("sae_seed") != 0
            or plan.get("tap") != "resid_pre_b8" or plan.get("concept") != "cCRE_PLS"
            or plan.get("target_feature") != 199 or plan.get("control_feature") != 4247
            or plan.get("match_sign") != 1
            or plan.get("intervention_scope") != "only cCRE_PLS-positive 128-bp bins"
            or plan.get("baseline_scope") != "SAE reconstruction on PLS bins; original tower input elsewhere"
            or plan.get("injection_rule") != "max(existing SAE code, dev positive-code p95) at PLS bins"
            or tuple(plan["primary_heads"] + plan["off_target_heads"]) != HEADS):
        raise ValueError("Unexpected local PLS protocol")
    if not 0 <= window_index < len(plan["windows"]):
        raise IndexError("Window index outside frozen local plan")
    parent = parent_override or Path(plan["parent_plan_path"])
    if sha256(parent) != plan["parent_plan_sha256"]:
        raise ValueError("Frozen parent selection changed")
    paths = {name: Path(value) for name, value in plan["source_paths"].items()}
    for name, relocated in (source_overrides or {}).items():
        if name not in paths:
            raise ValueError(f"Unknown source override: {name}")
        paths[name] = relocated
    for name, source in paths.items():
        if sha256(source) != plan["source_sha256"][name]:
            raise ValueError(f"Frozen source changed: {name}")
    window = plan["windows"][window_index]
    if (window["mode"] != "paired_local" or window["positive_bins"] <= 0
            or int(window["end"]) - int(window["start"]) != 1_048_576):
        raise ValueError("Window does not have a positive-bin local intervention")
    return plan, window, paths


def load_window_data(window: dict, paths: dict[str, Path],
                     acts_dir: Path, fasta_dir: Path):
    """Validate held-out status, complete bin alignment, and exact DNA span."""
    chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
    manifest = pd.read_parquet(paths["manifest"])
    held = manifest.loc[(manifest.split == "test") & (manifest.chrom == chrom)
                        & (manifest.win_start == start)]
    if len(held) != 1 or int(held.iloc[0].win_end) != end:
        raise ValueError("Window is not uniquely held out")
    annotations = pd.read_parquet(paths["annotations"])
    bins = annotations.loc[
        (annotations.split == "test") & (annotations.chrom == chrom)
        & (annotations.bin_start >= start) & (annotations.bin_start < end)
    ].sort_values("bin_start").reset_index(drop=True)
    expected = start + 128 * np.arange(8192, dtype=np.int64)
    if (not np.array_equal(bins.bin_start.to_numpy(dtype=np.int64), expected)
            or not bins.n_mask.to_numpy(dtype=bool).all()):
        raise ValueError("Complete, unmasked 128-bp annotation grid required")
    if bins.cCRE_PLS.isna().any() or not bins.cCRE_PLS.isin([False, True, 0, 1]).all():
        raise ValueError("PLS annotations must be complete binary labels")
    mask = checked_element_mask(bins.cCRE_PLS.to_numpy(dtype=bool), 8192)
    if int(mask.sum()) != window["positive_bins"]:
        raise ValueError("Frozen PLS-bin count changed")

    coords, coord_path = load_sampled_coordinates(acts_dir, "resid_pre_b8", chrom, start)
    if (len(coords) != 8192 or not (coords.split == "test").all()
            or not np.array_equal(coords.bin_start.to_numpy(dtype=np.int64), expected)):
        raise ValueError("Native sampled coordinates differ from annotation grid")
    index = json.loads(paths["activation_index"].read_text())
    relative = str(coord_path.relative_to(acts_dir))
    indexed = [shard["taps"]["resid_pre_b8"] for shard in index["shards"]
               if shard["split"] == "test" and "resid_pre_b8" in shard["taps"]
               and shard["taps"]["resid_pre_b8"]["coordinates"] == relative]
    coord_sha = sha256(coord_path)
    if len(indexed) != 1 or indexed[0]["coordinates_sha256"] != coord_sha:
        raise ValueError("Native coordinate hash differs from activation index")

    from alphagenome_pytorch.utils import sequence_to_onehot
    sequence = chrom_sequence(fasta_dir, chrom)[start:end]
    if len(sequence) != 1_048_576:
        raise ValueError("FASTA window length differs")
    onehot = sequence_to_onehot(sequence).astype(np.float32)
    if onehot.shape != (1_048_576, 4) or not onehot.sum(-1).reshape(8192, 128).all():
        raise ValueError("Ambiguous or malformed sequence")
    return expected, bins, mask, coord_sha, onehot


def read_head_summaries(output, label: str, arrays: dict[str, np.ndarray]) -> None:
    """Save 128-bp means and across-bin track means for each predeclared head."""
    for head in HEADS:
        value = output[head]
        if (not isinstance(value, dict) or 128 not in value
                or value[128].ndim != 3 or value[128].shape[:2] != (1, 8192)
                or not bool(torch.isfinite(value[128]).all())):
            raise ValueError(f"Invalid 128-bp {head} readout")
        full = value[128][0].float()
        arrays[f"{head}_{label}"] = full.mean(dim=-1).cpu().numpy()
        arrays[f"{head}_{label}_track_mean"] = full.mean(dim=0).cpu().numpy()


def predict_with_tower_input(model, tap, tensor, input_tensor, organism,
                             forward_args, label, arrays) -> None:
    """Run one full-model prediction after replacing the tower-block input."""
    with torch.inference_mode(), FullStreamTower(model, tap, run_replacement(tensor)):
        prediction = model.predict(input_tensor, organism, **forward_args)
    read_head_summaries(prediction, label, arrays)
    del prediction


def write_result(out: Path, arrays: dict | None, bins: pd.DataFrame, receipt: dict) -> None:
    """Create either measured readouts or an explicit, immutable skip receipt."""
    out.mkdir(parents=True, exist_ok=False)
    if arrays is not None:
        temporary = out / "readouts.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(out / "readouts.npz")
    bins[["chrom", "bin_start", "n_mask", "cCRE_PLS"]].to_parquet(
        out / "window_annotations.parquet", index=False
    )
    shutil.copy2(__file__, out / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name("fullstream_local_edit.py"),
                 out / "fullstream_local_edit.py")
    receipt["readouts_sha256"] = sha256(out / "readouts.npz") if arrays is not None else None
    receipt_tmp = out / "receipt.json.tmp"
    with receipt_tmp.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt_tmp.replace(out / "receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path,
                        help="Relocated byte-identical v1 plan (for account migration)")
    parser.add_argument("--source-override", action="append", default=[], metavar="NAME=PATH",
                        help="Relocated byte-identical frozen source; repeat as needed")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    overrides = {}
    for item in args.source_override:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in overrides:
            raise ValueError(f"Expected a unique NAME=PATH source override, got {item!r}")
        overrides[name] = Path(value)
    plan, window, paths = load_frozen_plan(
        args.plan, args.window_index, args.parent_plan, overrides
    )
    chrom, start = window["chrom"], int(window["start"])
    out = args.out_root / f"{args.window_index:02d}_paired_local_{chrom}_{start}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    expected, bins, mask, coord_sha, onehot = load_window_data(
        window, paths, args.acts_dir, args.fasta_dir
    )

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    tap = TAPS["resid_pre_b8"]
    if tap.kind != "tower" or tap.bin_bp != 128:
        raise ValueError("Expected a 128-bp tower-block input")
    sae = BorzoiSAE.from_checkpoint(paths["sae"], device="cuda")
    if sae.d_in != tap.channels or sae.hidden <= 4247:
        raise ValueError("SAE checkpoint dimensions differ from plan")
    model = AlphaGenome.from_pretrained(
        paths["weights"], dtype_policy=DtypePolicy.mixed_precision(), device="cuda"
    )
    model.eval().requires_grad_(False)
    input_tensor = torch.from_numpy(onehot)[None].to("cuda")
    organism = torch.tensor([0], device="cuda")
    forward_args = {"resolutions": (128,), "heads": HEADS, "channels_last": True}
    arrays: dict[str, np.ndarray] = {
        "bin_start": expected, "concept_label": mask, "intervention_mask": mask.copy(),
    }
    captured = []
    started = time.monotonic()

    def capture(value):
        captured.append(np.array(value, dtype=np.float32, copy=True))
        return value

    with torch.inference_mode(), FullStreamTower(model, tap, capture):
        original = model.predict(input_tensor, organism, **forward_args)
    if len(captured) != 1 or captured[0].shape != (8192, tap.channels):
        raise ValueError("Unexpected full-stream tap capture")
    read_head_summaries(original, "original", arrays)
    del original
    activation = torch.from_numpy(captured[0])[None].to("cuda")
    del captured
    levels = plan["calibration_levels"]
    ablation_baseline, ablation_target, ablation_control, ablation_counts = local_edit_triplet(
        sae, activation, mask, 199, 4247, "ablate",
        levels["target"], levels["control"], args.batch_size,
    )
    receipt = {
        "format": "fullstream_pls_local_paired_window_v2",
        "interpretation": plan["interpretation"],
        "plan_sha256": sha256(args.plan),
        "worker_sha256": sha256(Path(__file__)),
        "local_edit_sha256": sha256(Path(__file__).with_name("fullstream_local_edit.py")),
        "fold": plan["fold"], "sae_seed": plan["sae_seed"], "tap": plan["tap"],
        "tap_route": "tower_block_input_before_pair_update_mha_residual_mlp",
        "window_index": args.window_index, "window": window,
        "target_feature": 199, "control_feature": 4247,
        "injection_levels": levels, "injection_rule": plan["injection_rule"],
        "intervention_scope": plan["intervention_scope"],
        "baseline_scope": plan["baseline_scope"],
        "n_native_positions": len(mask), "n_intervened_bins": int(mask.sum()),
        "heads_primary": plan["primary_heads"],
        "heads_off_target": plan["off_target_heads"],
        "source_sha256": plan["source_sha256"],
        "source_paths_used": {name: str(path) for name, path in paths.items()},
        "parent_plan_path_used": str(args.parent_plan or plan["parent_plan_path"]),
        "sampled_coordinates_sha256": coord_sha,
    }
    if not paired_ablation_is_measurable(ablation_counts):
        receipt.update({
            "status": "unmeasurable_local_ablation",
            "reason": "Target or matched control has no effective activation change on PLS bins",
            "code_counts": {"ablate": ablation_counts},
            "elapsed_seconds": time.monotonic() - started,
        })
        write_result(out, None, bins, receipt)
        print(json.dumps({"out": str(out), "status": receipt["status"],
                          "code_counts": receipt["code_counts"],
                          "receipt_sha256": sha256(out / "receipt.json")}))
        return

    code_counts = {"ablate": ablation_counts}
    first_baseline = None
    for mode in ("ablate", "inject"):
        if mode == "ablate":
            baseline, target, control = ablation_baseline, ablation_target, ablation_control
        else:
            baseline, target, control, code_counts[mode] = local_edit_triplet(
                sae, activation, mask, 199, 4247, mode,
                levels["target"], levels["control"], args.batch_size,
            )
        if first_baseline is None:
            first_baseline = baseline.detach().clone()
            predict_with_tower_input(model, tap, baseline, input_tensor, organism,
                                     forward_args, "reconstruction", arrays)
        elif not torch.equal(first_baseline, baseline):
            raise ValueError("Ablation and injection do not share one baseline")
        for name, edited in (("target", target), ("control", control)):
            predict_with_tower_input(model, tap, edited, input_tensor, organism,
                                     forward_args, f"{mode}_{name}", arrays)
        if mode == "ablate":
            del ablation_baseline, ablation_target, ablation_control
        del baseline, target, control
        torch.cuda.empty_cache()
    del first_baseline, activation
    effects = {mode: summarize_local_effects(arrays, HEADS, mask, mode)
               for mode in ("ablate", "inject")}
    receipt.update({"status": "measured", "code_counts": code_counts,
                    "effects": effects, "elapsed_seconds": time.monotonic() - started})
    write_result(out, arrays, bins, receipt)
    print(json.dumps({"out": str(out), "receipt_sha256": sha256(out / "receipt.json"),
                      "readouts_sha256": receipt["readouts_sha256"],
                      "n_intervened_bins": int(mask.sum()), "code_counts": code_counts}))


if __name__ == "__main__":
    main()
