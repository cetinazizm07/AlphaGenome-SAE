#!/usr/bin/env python3
"""Resume-safe, sequential GPU runner for a frozen exploratory TSS cohort.

Rank zero is an already backed technical pilot and is never run again here.
Every later gene has an immutable plan, unique output, log and checksum
receipt. The manager stops at the first failure instead of launching duplicate
work or silently skipping a bad result.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from plan_fullstream_cohort import sha256


def checked_output(path: Path, record: dict) -> dict:
    """Validate a completed per-gene receipt before treating it as reusable."""
    receipt_path = path / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Existing output lacks receipt: {path}")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("format") != "fullstream_gene_tss_injection_pilot_v1"
            or receipt.get("plan_sha256") != record["plan_sha256"]
            or receipt.get("gene", {}).get("gene_id") != record["gene_id"]
            or receipt.get("status") not in {"measured", "unmeasurable"}):
        raise ValueError(f"Existing output is not the frozen gene result: {path}")
    readout = path / "readouts.npz"
    expected = receipt.get("readouts_sha256")
    if (expected is None) != (not readout.exists()):
        raise ValueError(f"Readout presence differs from receipt: {path}")
    if expected is not None and sha256(readout) != expected:
        raise ValueError(f"Readout checksum differs from receipt: {path}")
    if sha256(path / "plan.json") != record["plan_sha256"]:
        raise ValueError(f"Output plan differs from frozen input: {path}")
    return {"status": receipt["status"], "receipt_sha256": sha256(receipt_path),
            "readouts_sha256": expected, "effects": receipt.get("effects")}


def atomic_status(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(cohort: Path, out_root: Path, acts_dir: Path, fasta_dir: Path,
        worker: Path) -> dict:
    manifest_path = cohort / "cohort_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "exploratory_fold1_b8_gene_tss_cohort_v1"
            or manifest.get("fold") != "fold1" or manifest.get("tap") != "resid_pre_b8"
            or manifest.get("sae_seed") != 0
            or manifest["rank_cutoff_exclusive"] != len(manifest["records"])):
        raise ValueError("Unexpected frozen TSS cohort manifest")
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manager.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {
            "format": "exploratory_gene_tss_cohort_status_v1",
            "manifest_sha256": sha256(manifest_path),
            "worker_sha256": sha256(worker),
            "records": [],
        }
        for record in manifest["records"][1:]:
            rank = int(record["rank"])
            if rank < 1 or rank >= manifest["rank_cutoff_exclusive"]:
                raise ValueError("Malformed cohort rank")
            plan = cohort / record["plan_file"]
            if sha256(plan) != record["plan_sha256"]:
                raise ValueError(f"Frozen plan checksum differs at rank {rank}")
            output = out_root / f"rank{rank:02d}"
            if not output.exists():
                command = [sys.executable, str(worker), "--plan", str(plan),
                           "--out", str(output), "--acts-dir", str(acts_dir),
                           "--fasta-dir", str(fasta_dir)]
                log = out_root / f"rank{rank:02d}.log"
                if log.exists():
                    raise ValueError(f"Log exists without output; inspect before retry: {log}")
                with log.open("x") as stream:
                    started = time.time()
                    completed = subprocess.run(command, stdout=stream,
                                               stderr=subprocess.STDOUT, check=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                if completed.returncode:
                    raise RuntimeError(
                        f"Rank {rank} failed with exit {completed.returncode}; "
                        f"inspect {log} and {output} before any resume"
                    )
                elapsed = time.time() - started
            else:
                elapsed = None
            checked = checked_output(output, record)
            result["records"].append({
                "rank": rank, "gene_id": record["gene_id"],
                "gene_name": record["gene_name"], "output": str(output),
                "elapsed_wall_seconds": elapsed, **checked,
            })
            atomic_status(out_root / "campaign_status.json", result)
        return {"out": str(out_root), "completed": len(result["records"]),
                "expected": len(manifest["records"]) - 1,
                "status_sha256": sha256(out_root / "campaign_status.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name(
        "run_gene_tss_injection_pilot.py"))
    args = parser.parse_args()
    print(json.dumps(run(args.cohort, args.out_root, args.acts_dir,
                         args.fasta_dir, args.worker)))


if __name__ == "__main__":
    main()
