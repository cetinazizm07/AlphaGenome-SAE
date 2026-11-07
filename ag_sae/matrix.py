"""Matrix campaign orchestrator for AlphaGenome-SAE.

Iterates over layers and training seeds for a genomic fold (e.g. Fold 1's 9 configurations:
resid_pre_b8, resid_pre_b4, resid_pre_b0 x seed0, seed1, seed2) with complete process isolation
to prevent GPU memory leaks, automatic checkpoint resumption, and checksum-verified Google Drive syncing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LAYERS: tuple[str, ...] = ("resid_pre_b8", "resid_pre_b4", "resid_pre_b0")
DEFAULT_SEEDS: tuple[str, ...] = ("seed0", "seed1", "seed2")


def is_valid_json_file(path: Path | str | None) -> bool:
    """Return True if path points to an existing, non-empty, parseable JSON file."""
    if path is None:
        return False
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return False
    try:
        with open(target, "rb") as f:
            json.load(f)
        return True
    except Exception:
        return False



def run_matrix_campaign(
    fold: str = "fold1",
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    seeds: tuple[str, ...] = DEFAULT_SEEDS,
    base_out_dir: Path = Path("/mnt/ag/analysis_runs/campaign_matrix"),
    drive_remote: str = "gdrive:alphagenome_sae/campaign_matrix",
    dry_run: bool = False,
    sae_pool: str = "sae",
    max_batch: int = 3,
) -> None:
    """Execute evaluation matrix with process isolation and automatic resumption."""
    base_out_dir.mkdir(parents=True, exist_ok=True)
    configs = [(l, s) for l in layers for s in seeds]
    total = len(configs)

    print("=" * 80)
    print(f"STARTING CAMPAIGN MATRIX: {fold.upper()} | {len(layers)} Layers x {len(seeds)} Seeds ({total} Configs)")
    print(f"Output directory: {base_out_dir}")
    print(f"Drive destination: {drive_remote}")
    print("=" * 80)

    t_matrix_start = time.time()
    completed_count = 0

    for step, (layer, seed) in enumerate(configs, 1):
        config_name = f"{fold}_{layer}_{seed}"
        out_dir = base_out_dir / config_name
        triad_file = out_dir / "regulatory_triad_summary.json"
        gene_file = out_dir / "gene_level_statistical_summary.json"

        print(f"\n>>> [STEP {step}/{total}] Configuration: {config_name}")

        if is_valid_json_file(triad_file) and is_valid_json_file(gene_file):
            if not dry_run and drive_remote:
                target_drive_path = f"{drive_remote.rstrip('/')}/{config_name}/"
                sync_cmd = ["rclone", "copy", f"{out_dir}/", target_drive_path, "--checksum", "-q"]
                sync_res = subprocess.run(sync_cmd, capture_output=True, text=True)
                if sync_res.returncode != 0:
                    print(f"\n[ERROR] Google Drive sync failed for completed config {config_name} with returncode {sync_res.returncode}: {sync_res.stderr.strip()}", file=sys.stderr)
                    sys.exit(sync_res.returncode)
            print(f"  [SKIPPED] {config_name} is already complete and verified on disk & remote.")
            completed_count += 1
            continue

        cmd = [
            sys.executable, "-u", "-m", "ag_sae", "intervene",
            "--fold", fold,
            "--layer", layer,
            "--seed", seed,
            "--sae-pool", sae_pool,
            "--max-batch", str(max_batch),
            "--out", str(out_dir),
        ]

        if dry_run:
            print(f"  [DRY-RUN] Would execute: {' '.join(cmd)}")
            continue

        t0 = time.time()
        print(f"  Executing subprocess for {config_name}...")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n[ERROR] Step {step}/{total} failed with returncode {result.returncode} for {config_name}!", file=sys.stderr)
            sys.exit(result.returncode)

        if not is_valid_json_file(triad_file) or not is_valid_json_file(gene_file):
            print(f"\n[ERROR] Step {step}/{total} finished but output files are missing or corrupt on disk for {config_name}!", file=sys.stderr)
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"  [SUCCESS] Finished {config_name} in {elapsed:.1f}s!")

        # Sync to Google Drive with checksum verification
        if drive_remote:
            target_drive_path = f"{drive_remote.rstrip('/')}/{config_name}/"
            print(f"  Syncing to Google Drive ({target_drive_path})...")
            sync_cmd = ["rclone", "copy", f"{out_dir}/", target_drive_path, "--checksum", "-q"]
            sync_res = subprocess.run(sync_cmd, capture_output=True, text=True)
            if sync_res.returncode != 0:
                print(f"\n[ERROR] Google Drive sync failed for {config_name} with returncode {sync_res.returncode}: {sync_res.stderr.strip()}", file=sys.stderr)
                sys.exit(sync_res.returncode)

        completed_count += 1

    total_time = time.time() - t_matrix_start
    print("\n" + "=" * 80)
    print(f"CAMPAIGN MATRIX COMPLETE: {completed_count}/{total} configurations verified in {total_time:.1f}s!")
    print("=" * 80 + "\n")


def cmd_matrix(argv: list[str] | None = None) -> int:
    """CLI dispatcher for ag_sae matrix."""
    parser = argparse.ArgumentParser(description="AlphaGenome-SAE Campaign Matrix CLI")
    parser.add_argument("--fold", type=str, default="fold1", help="Genomic fold (fold0, fold1, fold2, fold3)")
    parser.add_argument("--layers", type=str, default="resid_pre_b8,resid_pre_b4,resid_pre_b0", help="Comma-separated layers")
    parser.add_argument("--seeds", type=str, default="seed0,seed1,seed2", help="Comma-separated seeds")
    parser.add_argument("--sae-pool", type=str, default="sae",
                        help="SAE directory under /mnt/ag (sae, sae_k64, sae_v2, ...)")
    parser.add_argument("--max-batch", type=int, default=3,
                        help="conditions per forward; 3-4 on an 80 GB A100, 1-2 on 40 GB")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default /mnt/ag/analysis_runs/campaign_matrix[_<pool>]")
    parser.add_argument("--drive-remote", type=str, default=None,
                        help="default gdrive:alphagenome_sae/campaign_matrix[_<pool>]")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args(argv)
    layers = tuple(p.strip() for p in args.layers.split(",") if p.strip())
    seeds = tuple(f"seed{p.strip().removeprefix('seed')}" for p in args.seeds.split(",") if p.strip())
    # Matches the triad directory that motif_intervene.resolve_ctcf_feature reads.
    name = "campaign_matrix" if args.sae_pool == "sae" else f"campaign_matrix_{args.sae_pool}"

    run_matrix_campaign(
        fold=args.fold,
        layers=layers,
        seeds=seeds,
        base_out_dir=args.out_dir or Path(f"/mnt/ag/analysis_runs/{name}"),
        drive_remote=args.drive_remote or f"gdrive:alphagenome_sae/{name}",
        dry_run=args.dry_run,
        sae_pool=args.sae_pool,
        max_batch=args.max_batch,
    )
    return 0
