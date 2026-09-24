#!/usr/bin/env python3
"""Precommit one held-out gene/TSS for a single-bin SAE injection pilot.

Selection uses only GENCODE annotation, the existing held-out manifest and
binary concept labels. It never reads model outputs or gene-expression values.
The same rule can be rerun from hashed inputs on another cloud account.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plan_fullstream_cohort import sha256


SELECTION_SEED = "fold1-b8-tss-injection-v1"
WINDOW_BP = 1_048_576
BIN_BP = 128
EDGE_MARGIN_BP = 131_072


def gtf_attributes(text: str) -> dict[str, list[str]]:
    """Parse repeated GTF attributes, including multiple `tag` entries."""
    attributes: dict[str, list[str]] = {}
    for field in text.strip().split(";"):
        parts = field.strip().split(" ", 1)
        if len(parts) == 2:
            attributes.setdefault(parts[0], []).append(parts[1].strip().strip('"'))
    return attributes


def first(attributes: dict[str, list[str]], key: str) -> str | None:
    values = attributes.get(key, [])
    return values[0] if values else None


def read_mane_transcripts(gtf: Path, chromosomes: set[str]) -> list[dict]:
    """Read protein-coding MANE Select transcripts, with 0-based coordinates."""
    opener = gzip.open if gtf.suffix == ".gz" else open
    transcripts = []
    with opener(gtf, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if len(fields) != 9 or fields[2] != "transcript" or fields[0] not in chromosomes:
                continue
            attrs = gtf_attributes(fields[8])
            if ("MANE_Select" not in attrs.get("tag", [])
                    or first(attrs, "gene_type") != "protein_coding"
                    or first(attrs, "transcript_type") != "protein_coding"):
                continue
            strand = fields[6]
            if strand not in {"+", "-"}:
                continue
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            if not 0 <= start0 < end0:
                raise ValueError("Malformed GTF transcript coordinates")
            transcript_id, gene_id = first(attrs, "transcript_id"), first(attrs, "gene_id")
            if not transcript_id or not gene_id:
                raise ValueError("MANE transcript lacks stable IDs")
            transcripts.append({
                "chrom": fields[0], "start0": start0, "end0": end0,
                "tss0": start0 if strand == "+" else end0 - 1,
                "strand": strand, "gene_id": gene_id,
                "gene_name": first(attrs, "gene_name") or gene_id,
                "transcript_id": transcript_id,
            })
    return transcripts


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union exon intervals so overlapping exons are never double counted."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            raise ValueError("Empty or reversed exon interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def read_selected_exons(gtf: Path, transcript_ids: set[str]) -> dict[str, list[tuple[int, int]]]:
    """Second streaming pass reads exons only for eligible MANE transcripts."""
    opener = gzip.open if gtf.suffix == ".gz" else open
    exons: dict[str, list[tuple[int, int]]] = {name: [] for name in transcript_ids}
    with opener(gtf, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if len(fields) != 9 or fields[2] != "exon":
                continue
            transcript_id = first(gtf_attributes(fields[8]), "transcript_id")
            if transcript_id in exons:
                exons[transcript_id].append((int(fields[3]) - 1, int(fields[4])))
    return {name: merge_intervals(parts) for name, parts in exons.items()}


def exon_bin_weights(exons: list[tuple[int, int]], window_start: int) -> list[dict[str, int]]:
    """Number of annotated exon bases represented by each 128-bp output bin."""
    weights: dict[int, int] = {}
    for exon_start, exon_end in merge_intervals(exons):
        if exon_start < window_start or exon_end > window_start + WINDOW_BP:
            raise ValueError("Selected transcript has exons outside the 1-Mb window")
        for index in range((exon_start - window_start) // BIN_BP,
                           (exon_end - 1 - window_start) // BIN_BP + 1):
            bin_start = window_start + BIN_BP * index
            overlap = min(exon_end, bin_start + BIN_BP) - max(exon_start, bin_start)
            weights[index] = weights.get(index, 0) + overlap
    if any(not 0 < weight <= BIN_BP for weight in weights.values()):
        raise ValueError("Invalid exon overlap weighting")
    return [{"bin_index": index, "exon_bp": weights[index]} for index in sorted(weights)]


def complete_window_annotations(annotations: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Validate complete test grids once and retain only pre-existing labels."""
    required = {"split", "chrom", "bin_start", "n_mask", "cCRE_PLS"}
    if not required <= set(annotations):
        raise ValueError(f"Missing annotation columns: {sorted(required - set(annotations))}")
    by_chrom = {chrom: part.sort_values("bin_start") for chrom, part in
                annotations.loc[annotations.split == "test"].groupby("chrom")}
    grids = {}
    for row in manifest.loc[manifest.split == "test"].itertuples():
        start, end = int(row.win_start), int(row.win_end)
        if end - start != WINDOW_BP or row.chrom not in by_chrom:
            continue
        chrom_rows = by_chrom[row.chrom]
        starts = chrom_rows.bin_start.to_numpy(dtype=np.int64)
        left, right = starts.searchsorted([start, end])
        bins = chrom_rows.iloc[left:right]
        if not np.array_equal(bins.bin_start.to_numpy(dtype=np.int64),
                              start + BIN_BP * np.arange(WINDOW_BP // BIN_BP)):
            continue
        if not bins.n_mask.to_numpy(dtype=bool).all():
            continue
        grids[(row.chrom, start)] = {
            "end": end,
            "pls": bins.cCRE_PLS.to_numpy(dtype=bool),
        }
    return grids


def eligible_genes(gtf: Path, manifest: pd.DataFrame, annotations: pd.DataFrame) -> list[dict]:
    """Pick annotated, isolated transcript starts in held-out complete-grid windows."""
    grids = complete_window_annotations(annotations, manifest)
    transcripts = read_mane_transcripts(gtf, {chrom for chrom, _ in grids})
    preliminary = []
    for transcript in transcripts:
        for (chrom, window_start), grid in grids.items():
            if chrom != transcript["chrom"]:
                continue
            if not (window_start + EDGE_MARGIN_BP <= transcript["tss0"]
                    < grid["end"] - EDGE_MARGIN_BP
                    and window_start <= transcript["start0"]
                    and transcript["end0"] <= grid["end"]):
                continue
            tss_bin = (transcript["tss0"] - window_start) // BIN_BP
            if grid["pls"][tss_bin]:
                continue
            preliminary.append(dict(transcript, window_start=window_start,
                                    window_end=grid["end"], tss_bin_index=int(tss_bin)))
            break
    exons = read_selected_exons(gtf, {row["transcript_id"] for row in preliminary})
    eligible = []
    for row in preliminary:
        transcript_exons = exons[row["transcript_id"]]
        if not transcript_exons:
            continue
        weights = exon_bin_weights(transcript_exons, row["window_start"])
        if sum(item["exon_bp"] for item in weights) < 2_048 or len(weights) < 16:
            continue
        row["exons_0based_halfopen"] = transcript_exons
        row["exon_bin_weights"] = weights
        row["selection_hash"] = hashlib.sha256(
            f"{SELECTION_SEED}|{row['gene_id']}|{row['transcript_id']}".encode()
        ).hexdigest()
        eligible.append(row)
    eligible.sort(key=lambda row: (row["selection_hash"], row["gene_id"]))
    return eligible


def select_tracks(strand: str) -> dict[str, list[dict]]:
    """Freeze every matching, unmodified human RNA/CAGE/PRO-cap track."""
    from alphagenome_pytorch.named_outputs import TrackMetadataCatalog

    catalog = TrackMetadataCatalog.load_builtin("human")
    selected = {}
    for output_name in ("rna_seq", "cage", "procap"):
        tracks = []
        for track in catalog.get_tracks(output_name, organism=0):
            extras = track.extras
            if extras.get("strand") != strand or extras.get("genetically_modified"):
                continue
            if output_name == "rna_seq" and extras.get("assay_title") != "total RNA-seq":
                continue
            tracks.append({
                "index": int(track.track_index), "name": track.track_name,
                "biosample": extras.get("biosample_name"),
                "assay": extras.get("assay_title"), "strand": strand,
            })
        if not tracks or len({item["index"] for item in tracks}) != len(tracks):
            raise ValueError(f"No unique strand-matched {output_name} channels")
        selected[output_name] = tracks
    return selected


def candidate_at_rank(candidates: list[dict], rank: int) -> dict:
    """Select by a predeclared position, never by an intervention result."""
    if not 0 <= rank < len(candidates):
        raise ValueError(f"Gene rank {rank} is outside {len(candidates)} eligible genes")
    return candidates[rank]


def validated_context(parent_path: Path, gtf: Path) -> tuple[dict, list[dict]]:
    """Read the immutable inputs and compute the candidate order only once."""
    parent = json.loads(parent_path.read_text())
    if (parent.get("format") != "fullstream_pls_local_paired_plan_v2"
            or parent.get("fold") != "fold1" or parent.get("sae_seed") != 0
            or parent.get("tap") != "resid_pre_b8" or parent.get("target_feature") != 199
            or parent.get("control_feature") != 4247):
        raise ValueError("Unexpected frozen fold1/b8 source plan")
    paths = {name: Path(value) for name, value in parent["source_paths"].items()}
    if any(sha256(path) != parent["source_sha256"][name] for name, path in paths.items()):
        raise ValueError("Frozen source changed")
    manifest = pd.read_parquet(paths["manifest"])
    annotations = pd.read_parquet(paths["annotations"])
    candidates = eligible_genes(gtf, manifest, annotations)
    return parent, candidates


def plan_at_rank(parent_path: Path, gtf: Path, parent: dict,
                 candidates: list[dict], rank: int) -> dict:
    """Freeze one gene at a fixed rank in the outcome-blind candidate order."""
    selected = candidate_at_rank(candidates, rank)
    tracks = select_tracks(selected["strand"])
    return {
        "format": "fullstream_gene_tss_injection_plan_v1",
        "interpretation": (
            "Exploratory one-gene, one-bin model-prediction gain-of-function pilot; "
            "not biological expression measurement or population sufficiency proof."
        ),
        "parent_plan_path": str(parent_path.resolve()),
        "parent_plan_sha256": sha256(parent_path),
        "fold": "fold1", "sae_seed": 0, "tap": "resid_pre_b8",
        "split": "test", "target_feature": 199, "control_feature": 4247,
        "target_dev_positive_p95": parent["calibration_levels"]["target"],
        "control_dev_positive_p95": parent["calibration_levels"]["control"],
        "selection_seed": SELECTION_SEED,
        "selection_rank": rank,
        "selection_rule": (
            f"Rank {rank} by ascending SHA256(seed|gene_id|MANE transcript_id) "
            "among protein-coding MANE "
            "transcripts entirely inside a complete-grid fold1 test window, TSS at least "
            "131072 bp from edges, TSS defined by the MANE transcript and not "
            "labeled cCRE_PLS, and >=2048 exonic bp across >=16 bins. No model outcome used."
        ),
        "n_eligible_genes": len(candidates), "gene": selected,
        "tracks": tracks,
        "track_rule": (
            "All unmodified human strand-matched total RNA-seq tracks for primary "
            "gene-exon coverage; all strand-matched CAGE/PRO-cap tracks secondary."
        ),
        "primary_rna_track_rule": (
            "Among the frozen strand-matched total RNA-seq tracks, report the track "
            "with the highest unmodified original exon-weighted prediction; break "
            "ties by frozen track-list order. Never use intervention outcomes "
            "to select a tissue. Also retain every track and the all-track median."
        ),
        "source_paths": parent["source_paths"],
        "source_sha256": parent["source_sha256"],
        "gtf_path": str(gtf.resolve()), "gtf_sha256": sha256(gtf),
        "gtf_source_url": (
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
            "release_50/gencode.v50.basic.annotation.gtf.gz"
        ),
    }


def build_plan(parent_path: Path, gtf: Path, rank: int = 0) -> dict:
    """Standalone single-gene planner, useful for a small technical pilot."""
    parent, candidates = validated_context(parent_path, gtf)
    return plan_at_rank(parent_path, gtf, parent, candidates, rank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0,
                        help="Zero-based position in the frozen, outcome-blind gene order")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    plan = build_plan(args.parent_plan, args.gtf, args.rank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    gene = plan["gene"]
    print(json.dumps({"plan": str(args.out), "sha256": sha256(args.out),
                      "gene": gene["gene_name"], "gene_id": gene["gene_id"],
                      "chrom": gene["chrom"], "tss0": gene["tss0"],
                      "n_eligible": plan["n_eligible_genes"], "rank": args.rank,
                      "rna_tracks": len(plan["tracks"]["rna_seq"])}))


if __name__ == "__main__":
    main()
