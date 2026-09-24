#!/usr/bin/env python3
"""Run and validate only the immutable, matched non-pilot PLS cohort."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from plan_fullstream_cohort import sha256


HEADS = ("dnase", "atac", "cage", "rna_seq", "procap", "chip_tf", "chip_histone")


def validate_output(out: Path, cohort: dict, entry: dict) -> dict:
    receipt_path, readouts_path = out / "receipt.json", out / "readouts.npz"
    if not receipt_path.is_file() or not readouts_path.is_file():
        raise ValueError(f"Incomplete output at {out}")
    receipt = json.loads(receipt_path.read_text())
    expected_bins = int(entry["target_active_bins"])
    expected_window = entry["window"]
    if (receipt.get("status") != "measured"
            or receipt.get("format") != "fullstream_pls_local_ablation_window_v6"
            or receipt.get("plan_sha256") != cohort["plan_sha256"]
            or receipt.get("window_index") != entry["window_index"]
            or receipt.get("window") != expected_window
            or receipt.get("target_feature") != cohort["target_feature"]
            or receipt.get("control_feature") != entry["control_feature"]
            or receipt.get("n_intervened_bins") != expected_bins
            or receipt.get("n_concept_positive_bins") != expected_window["positive_bins"]
            or receipt.get("window_control_audit_sha256") != entry["control_audit_sha256"]):
        raise ValueError(f"Receipt does not match frozen cohort plan at window {entry['window_index']}")
    counts = receipt["code_counts"]
    norms = receipt["tap_intervention"]
    ratio = (norms["comparator_mean_l2_per_intervened_bin"]
             / norms["target_mean_l2_per_intervened_bin"])
    if (counts["target_changed_on_element"] != expected_bins
            or counts["control_changed_on_element"] != expected_bins
            or counts["target_activation_changed_on_element"] != expected_bins
            or counts["control_activation_changed_on_element"] != expected_bins
            or norms["coactive_bins"] != expected_bins
            or not 0.75 <= ratio <= 1.25):
        raise ValueError(f"Intervention support or realized dose mismatch at window {entry['window_index']}")
    if sha256(readouts_path) != receipt.get("readouts_sha256"):
        raise ValueError(f"Readout checksum mismatch at window {entry['window_index']}")
    if sha256(Path(entry["control_audit_path"])) != entry["control_audit_sha256"]:
        raise ValueError(f"Control audit checksum mismatch at window {entry['window_index']}")
    data = np.load(readouts_path, allow_pickle=False)
    support = data["intervention_mask"].astype(bool)
    concept = data["concept_label"].astype(bool)
    if (support.shape != (8192,) or concept.shape != (8192,)
            or int(support.sum()) != expected_bins
            or int(concept.sum()) != int(expected_window["positive_bins"])
            or np.any(support & ~concept)):
        raise ValueError(f"Saved support mask is not a subset of PLS bins at window {entry['window_index']}")
    for head in HEADS:
        if not np.array_equal(data[f"{head}_baseline"], data[f"{head}_original"]):
            raise ValueError(f"Raw-model baseline changed for {head} at window {entry['window_index']}")
        for arm in ("original", "baseline", "ablate_target", "ablate_control"):
            value = data[f"{head}_{arm}"]
            if value.shape != (8192,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {head}/{arm} track at window {entry['window_index']}")
    return {
        "window_index": entry["window_index"], "window": expected_window,
        "control_feature": entry["control_feature"],
        "target_active_bins": expected_bins,
        "realized_control_to_target_dose_ratio": float(ratio),
        "receipt_sha256": sha256(receipt_path),
        "readouts_sha256": sha256(readouts_path),
    }


def run_cohort(cohort_path: Path, plan_path: Path, runner: Path,
               local_edit: Path, tower_hook: Path, out_root: Path,
               acts: Path, fasta: Path, batch_size: int = 256) -> dict:
    cohort = json.loads(cohort_path.read_text())
    if (cohort.get("format") != "residual_preserving_fullstream_pls_ablation_cohort_v1"
            or cohort.get("status") != "frozen_before_cohort_output_inference"
            or sha256(plan_path) != cohort["plan_sha256"]
            or sha256(runner) != cohort["code_sha256"]["runner"]
            or sha256(local_edit) != cohort["code_sha256"]["local_edit"]
            or sha256(tower_hook) != cohort["code_sha256"]["tower_hook"]
            or sha256(Path(__file__)) != cohort["code_sha256"]["manager"]):
        raise ValueError("Frozen cohort plan or executable code changed")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)
    lock_path = out_root / "manager.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another cohort manager owns this output directory") from error
        validated = []
        for entry in cohort["windows"]:
            window = entry["window"]
            index = int(entry["window_index"])
            name = f"{index:02d}_{window['chrom']}_{window['start']}"
            out = out_root / name
            if not out.exists():
                command = [
                    sys.executable, str(runner), "--plan", str(plan_path),
                    "--window-index", str(index), "--control-audit",
                    entry["control_audit_path"], "--acts-dir", str(acts),
                    "--fasta-dir", str(fasta), "--out", str(out),
                    "--batch-size", str(batch_size),
                ]
                completed = subprocess.run(command, text=True, capture_output=True)
                log = out_root / "logs" / f"{name}.log"
                log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Window {index} failed ({completed.returncode}); see {log}"
                    )
            record = validate_output(out, cohort, entry)
            validated.append(record)
            print(json.dumps({"validated": len(validated), "total": len(cohort["windows"]),
                              **record}), flush=True)
        summary = {
            "format": "residual_preserving_fullstream_pls_ablation_run_v1",
            "status": "all_frozen_windows_validated",
            "cohort_plan_path": str(cohort_path),
            "cohort_plan_sha256": sha256(cohort_path),
            "original_plan_sha256": cohort["plan_sha256"],
            "manager_sha256": sha256(Path(__file__)),
            "n_expected": len(cohort["windows"]),
            "n_validated": len(validated),
            "windows": validated,
        }
        status_path = out_root / "run_status.json"
        if status_path.exists():
            existing = json.loads(status_path.read_text())
            if existing != summary:
                raise ValueError("Existing cohort status differs; refusing to overwrite")
        else:
            with status_path.open("x") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        return {"out": str(out_root), "validated": len(validated),
                "status_sha256": sha256(status_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-plan", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--local-edit", type=Path, required=True)
    parser.add_argument("--tower-hook", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run_cohort(
        args.cohort_plan, args.plan, args.runner, args.local_edit,
        args.tower_hook, args.out_root, args.acts_dir, args.fasta_dir,
        args.batch_size,
    )))


if __name__ == "__main__":
    main()
