#!/usr/bin/env python3
"""Freeze a small, outcome-blind TSS cohort before further GPU inference.

Rank zero is the immutable ACSBG2 technical pilot. Ranks one through cutoff-1
are selected from the same SHA-256 candidate order without reading model
predictions. This is an exploratory technical cohort, not a preregistered
population-level causal study.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from plan_fullstream_cohort import sha256
from plan_gene_tss_injection import plan_at_rank, validated_context


def atomic_json(path: Path, value: dict) -> None:
    """Persist a provenance record without leaving a partial final JSON file."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze(parent_plan: Path, gtf: Path, pilot_plan: Path, pilot_receipt: Path,
           out: Path, cutoff: int) -> dict:
    """Create all future rank plans and a manifest before any new outcomes."""
    if out.exists() or cutoff < 2:
        raise ValueError("Output must be new and cutoff must include the pilot plus another gene")
    parent, candidates = validated_context(parent_plan, gtf)
    if cutoff > len(candidates):
        raise ValueError(f"Cutoff {cutoff} exceeds {len(candidates)} eligible genes")
    pilot = json.loads(pilot_plan.read_text())
    receipt = json.loads(pilot_receipt.read_text())
    if (pilot["gene"]["gene_id"] != candidates[0]["gene_id"]
            or pilot["gene"]["transcript_id"] != candidates[0]["transcript_id"]
            or pilot["parent_plan_sha256"] != sha256(parent_plan)
            or pilot["gtf_sha256"] != sha256(gtf)
            or receipt["status"] != "measured"
            or receipt["plan_sha256"] != sha256(pilot_plan)):
        raise ValueError("Rank-zero pilot does not match the frozen candidate order")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".staging-", dir=out.parent))
    records = [{
        "rank": 0, "gene_id": candidates[0]["gene_id"],
        "gene_name": candidates[0]["gene_name"],
        "transcript_id": candidates[0]["transcript_id"],
        "status": "existing_technical_pilot",
        "plan_sha256": sha256(pilot_plan),
        "receipt_sha256": sha256(pilot_receipt),
        "readouts_sha256": receipt["readouts_sha256"],
    }]
    for rank in range(1, cutoff):
        plan = plan_at_rank(parent_plan, gtf, parent, candidates, rank)
        name = f"rank{rank:02d}_plan.json"
        path = staging / name
        atomic_json(path, plan)
        records.append({
            "rank": rank, "gene_id": plan["gene"]["gene_id"],
            "gene_name": plan["gene"]["gene_name"],
            "transcript_id": plan["gene"]["transcript_id"],
            "status": "planned", "plan_file": name,
            "plan_sha256": sha256(path),
        })
    manifest = {
        "format": "exploratory_fold1_b8_gene_tss_cohort_v1",
        "interpretation": (
            "Technical exploratory cohort; cutoff frozen after rank-zero pilot "
            "but before ranks one onward were run. No confirmatory p-value."
        ),
        "fold": "fold1", "tap": "resid_pre_b8", "sae_seed": 0,
        "rank_cutoff_exclusive": cutoff,
        "selection_rule": "First ranks in fixed SHA256 annotation-only candidate order",
        "primary_rna_track_rule": (
            "For each gene, select highest unmodified original exon-weighted "
            "strand-matched total RNA-seq track before looking at target/control effects"
        ),
        "n_eligible_genes": len(candidates),
        "parent_plan_sha256": sha256(parent_plan),
        "gtf_sha256": sha256(gtf),
        "source_sha256": parent["source_sha256"],
        "records": records,
    }
    atomic_json(staging / "cohort_manifest.json", manifest)
    staging.rename(out)
    return {"out": str(out), "manifest_sha256": sha256(out / "cohort_manifest.json"),
            "cutoff": cutoff, "genes": [row["gene_name"] for row in records]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--pilot-plan", type=Path, required=True)
    parser.add_argument("--pilot-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(freeze(args.parent_plan, args.gtf, args.pilot_plan,
                            args.pilot_receipt, args.out, args.cutoff)))


if __name__ == "__main__":
    main()
