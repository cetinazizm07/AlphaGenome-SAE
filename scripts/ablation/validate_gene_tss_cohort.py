#!/usr/bin/env python3
"""Independent receipt/checksum and paired-readout audit of the TSS cohort.

This checks the nine new rank outputs plus the pre-existing rank-zero pilot.
It produces descriptive, per-gene results only; correlated assay tracks and
different genes are not treated as interchangeable statistical replicates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gene_tss_readout import summarize_gene_effects
from plan_fullstream_cohort import sha256
from run_gene_tss_cohort import checked_output


def assert_close_record(a, b, path: str = "effects") -> None:
    """Compare nested readout summaries without hiding a field mismatch."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"Summary fields differ at {path}")
        for key in a:
            assert_close_record(a[key], b[key], path + "." + key)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not np.isclose(a, b, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Summary value differs at {path}: {a} vs {b}")
    elif a != b:
        raise ValueError(f"Summary value differs at {path}: {a!r} vs {b!r}")


def read_and_check(path: Path, plan_hash: str, status: dict | None = None) -> dict:
    """Recompute every per-gene effect from the archived track-level arrays."""
    plan_path, receipt_path, readout_path = (
        path / "plan.json", path / "receipt.json", path / "readouts.npz"
    )
    if sha256(plan_path) != plan_hash:
        raise ValueError(f"Plan hash mismatch: {path}")
    plan = json.loads(plan_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("status") != "measured"
            or receipt.get("plan_sha256") != plan_hash
            or receipt.get("readouts_sha256") != sha256(readout_path)
            or receipt.get("gene", {}).get("gene_id") != plan["gene"]["gene_id"]
            or receipt.get("n_intervened_bins") != 1):
        raise ValueError(f"Receipt, gene, or single-bin intervention differs: {path}")
    gene = plan["gene"]
    expected_bin = gene["window_start"] + 128 * gene["tss_bin_index"]
    if (receipt["tss_bin_start"] != expected_bin
            or not expected_bin <= gene["tss0"] < expected_bin + 128):
        raise ValueError(f"TSS coordinate mismatch: {path}")
    norms = receipt["tap_intervention_l2"]
    if not np.isclose(norms["target_l2"], norms["control_matched_l2"],
                      rtol=1e-4, atol=1e-5):
        raise ValueError(f"Target/control dose mismatch: {path}")
    counts = receipt["code_counts"]
    if (counts["element_bins"] != 1
            or counts["target_changed_on_element"] != 1
            or counts["control_changed_on_element"] != 1):
        raise ValueError(f"One-bin code edit did not occur: {path}")
    arrays = dict(np.load(readout_path, allow_pickle=False))
    recomputed = summarize_gene_effects(arrays, plan["tracks"])
    if "highest_original_expression_track" not in receipt["effects"]["rna_seq"]:
        # Rank zero was run with the earlier pilot code; never rewrite its
        # immutable receipt. Its top-track analysis remains explicitly post-hoc.
        del recomputed["rna_seq"]["highest_original_expression_track"]
        primary = None
    else:
        primary = recomputed["rna_seq"]["highest_original_expression_track"]
    assert_close_record(recomputed, receipt["effects"])
    for name, expected in (
        ("run_gene_tss_injection_pilot.py", receipt["worker_sha256"]),
        ("gene_tss_readout.py", receipt["readout_code_sha256"]),
        ("fullstream_local_edit.py", receipt["local_edit_code_sha256"]),
    ):
        if sha256(path / name) != expected:
            raise ValueError(f"Worker source hash differs: {path / name}")
    if status is not None:
        if status["receipt_sha256"] != sha256(receipt_path):
            raise ValueError(f"Campaign status does not match receipt: {path}")
        assert_close_record(status["effects"], receipt["effects"])
    return {
        "gene_id": gene["gene_id"], "gene_name": gene["gene_name"],
        "transcript_id": gene["transcript_id"], "strand": gene["strand"],
        "receipt_sha256": sha256(receipt_path),
        "readouts_sha256": sha256(readout_path),
        "primary_rna_track": primary,
        "all_track_median_target_log2_ratio": receipt["effects"]["rna_seq"][
            "target_median_log2_ratio"],
        "all_track_median_control_log2_ratio": receipt["effects"]["rna_seq"][
            "control_median_log2_ratio"],
        "all_track_reconstruction_median_abs_log2_ratio": receipt["effects"]["rna_seq"][
            "reconstruction_median_abs_log2_ratio"],
    }


def validate(cohort: Path, pilot: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    manifest_path = cohort / "plans" / "cohort_manifest.json"
    status_path = cohort / "runs" / "campaign_status.json"
    manifest, status = (json.loads(path.read_text()) for path in
                        (manifest_path, status_path))
    if (manifest["format"] != "exploratory_fold1_b8_gene_tss_cohort_v1"
            or status["manifest_sha256"] != sha256(manifest_path)
            or len(status["records"]) != len(manifest["records"]) - 1):
        raise ValueError("Incomplete or inconsistent cohort status")
    records = manifest["records"]
    rows = [read_and_check(pilot, records[0]["plan_sha256"])]
    rows[0].update({"rank": 0, "primary_track_rule_status": "post_hoc_pilot"})
    for record, state in zip(records[1:], status["records"], strict=True):
        if (record["rank"] != state["rank"]
                or record["gene_id"] != state["gene_id"]):
            raise ValueError("Rank or gene differs between plan and status")
        plan_path = cohort / "plans" / record["plan_file"]
        if sha256(plan_path) != record["plan_sha256"]:
            raise ValueError(f"Frozen plan hash differs for rank {record['rank']}")
        output = cohort / "runs" / f"rank{record['rank']:02d}"
        checked_output(output, record)
        row = read_and_check(output, record["plan_sha256"], state)
        row.update({"rank": record["rank"],
                    "primary_track_rule_status": "frozen_before_outcome"})
        rows.append(row)
    if len({row["gene_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate gene in frozen ranks")
    future = [row for row in rows if row["rank"] > 0]
    paired = [row["primary_rna_track"]["target_minus_control_log2_ratio"]
              for row in future]
    summary = {
        "format": "validated_exploratory_tss_cohort_v1",
        "interpretation": "Descriptive model-prediction interventions; no biological expression measurement or confirmatory statistical test",
        "manifest_sha256": sha256(manifest_path),
        "campaign_status_sha256": sha256(status_path),
        "n_genes": len(rows), "n_frozen_after_pilot": len(future),
        "new_gene_primary_paired_median_log2_ratio": float(np.median(paired)),
        "new_gene_primary_paired_positive_count": int(np.sum(np.asarray(paired) > 0)),
        "genes": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out)
    return {"out": str(out), "sha256": sha256(out), "n_genes": len(rows),
            "new_gene_primary_paired_median_log2_ratio": summary[
                "new_gene_primary_paired_median_log2_ratio"],
            "new_gene_primary_paired_positive_count": summary[
                "new_gene_primary_paired_positive_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.cohort, args.pilot, args.out)))


if __name__ == "__main__":
    main()
