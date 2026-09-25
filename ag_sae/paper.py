"""External 55-concept benchmark from Nair et al. (ICML 2026).

The benchmark is deliberately built into its own annotation matrix and panel.
It never changes the frozen v3 confirmatory hypotheses or the intervention
readout plan.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .config import BIN_BP, WIN
from .data import atomic_json, read_annotation, sha256
from .prepare.annotations import mark_bins, read_bed


PAPER_URL = "https://github.com/akira-nair/glm-sae-interpretability/blob/main/icml2026_manuscript.pdf"
REPO_URL = "https://github.com/akira-nair/glm-sae-interpretability"
DOI = "10.5281/zenodo.20185940"

# The eight Registry V4 classes already present in the frozen project matrix.
SHARED_COLUMNS = {
    "cCRE_CA": "cCRE_CA",
    "cCRE_CA_CTCF": "cCRE_CTCF",
    "cCRE_CA_H3K4me3": "cCRE_CA_H3K4me3",
    "cCRE_CA_TF": "cCRE_CA_TF",
    "cCRE_dELS": "cCRE_dELS",
    "cCRE_pELS": "cCRE_pELS",
    "cCRE_PLS": "cCRE_PLS",
    "cCRE_TF": "cCRE_TF",
}

GROUPS = {
    "regulatory_elements": [
        *SHARED_COLUMNS,
        "CpG_island", "super_enhancer_element", "super_enhancer",
        "typical_enhancer",
    ],
    "gene_structure": [
        "MANE_3UTR", "MANE_5UTR", "MANE_CDS", "MANE_TSS", "MANE_exon",
        "MANE_intron", "MANE_lncRNA_gene", "MANE_promoter",
        "MANE_protein_coding_gene", "MANE_snRNA_gene", "MANE_start_codon",
        "MANE_stop_codon", "polyA_signal", "polyA_site", "pseudo_polyA",
    ],
    "gene_body_elements": ["IG_C_gene", "IG_J_gene", "IG_V_gene"],
    "repetitive_elements": [
        "repeat_DNA", "repeat_LINE", "repeat_LTR", "repeat_low_complexity",
        "repeat_RC", "repeat_retroposon", "repeat_SINE", "repeat_satellite",
        "repeat_simple", "repeat_RNA", "repeat_rRNA", "repeat_tRNA",
        "repeat_scRNA", "repeat_snRNA", "repeat_srpRNA",
    ],
    "noncoding_RNA": ["miRNA", "misc_RNA", "piRNA", "rRNA", "scaRNA", "tRNA"],
    "conservation": ["PhastCons_conserved", "PhyloP_conserved"],
    "other": ["CRISPR_target", "selenocysteine"],
}

CONCEPTS = [concept for concepts in GROUPS.values() for concept in concepts]
EXTERNAL_CONCEPTS = [concept for concept in CONCEPTS if concept not in SHARED_COLUMNS]
SPARSE_EVENT_CONCEPTS = [
    "MANE_TSS", "MANE_start_codon", "MANE_stop_codon", "miRNA",
    "selenocysteine",
]

EVENT_ENRICHMENT = {
    "radius_bins": 2,
    "radius_bp": 2 * BIN_BP,
    "sigma_bins": 1.0,
    "sigma_bp": BIN_BP,
    "block_bp": WIN,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20260921,
    "confidence": 0.95,
    "selection_note": (
        "Fixed before benchmark execution. The paper's 10-bp radius is not "
        "resolvable on AlphaGenome's 128-bp SAE grid."
    ),
}

# The source paper reverse-complements negative-strand examples before Evo2
# evaluation.  This project currently scores the AlphaGenome 128-bp reference
# grid, so strand is collapsed into a single overlap label.  Recording the
# affected concepts prevents this adaptation from being mistaken for an exact
# replication of the paper's orientation-aware sequence sampling.
DIRECTIONAL_CONCEPTS = [
    *GROUPS["gene_structure"], *GROUPS["gene_body_elements"],
    *GROUPS["noncoding_RNA"], "selenocysteine",
]

SOURCES = {
    **{name: "ENCODE Registry V4" for name in SHARED_COLUMNS},
    "CpG_island": "UCSC CpG Islands",
    "super_enhancer_element": "SEdb 2.0",
    "super_enhancer": "SEdb 2.0",
    "typical_enhancer": "SEdb 2.0",
    **{name: "MANE v1.4" for name in GROUPS["gene_structure"] if name.startswith("MANE_")},
    "polyA_signal": "GENCODE v49 polyA annotation",
    "polyA_site": "GENCODE v49 polyA annotation",
    "pseudo_polyA": "GENCODE v49 polyA annotation",
    **{name: "GENCODE v49" for name in GROUPS["gene_body_elements"]},
    **{name: "RepeatMasker" for name in GROUPS["repetitive_elements"]},
    "miRNA": "GENCODE v49", "misc_RNA": "GENCODE v49",
    "piRNA": "piRBase v3.0", "rRNA": "GENCODE v49",
    "scaRNA": "GENCODE v49", "tRNA": "GENCODE v49 tRNA annotation",
    "PhastCons_conserved": "UCSC 100-way vertebrate PhastCons elements",
    "PhyloP_conserved": "UCSC 100-way vertebrate PhyloP elements",
    "CRISPR_target": "UCSC CRISPR Targets",
    "selenocysteine": "GENCODE v49",
}


def _validate_spec():
    if len(CONCEPTS) != 55 or len(set(CONCEPTS)) != 55:
        raise RuntimeError("Paper benchmark specification must contain 55 unique concepts")
    if len(SHARED_COLUMNS) != 8 or len(EXTERNAL_CONCEPTS) != 47:
        raise RuntimeError("Paper benchmark must contain 8 shared and 47 external concepts")
    if set(SOURCES) != set(CONCEPTS):
        raise RuntimeError("Every paper concept must have exactly one declared source")


_validate_spec()


def source_template():
    return {
        "schema_version": 1,
        "benchmark": "nair_icml2026_55_concepts",
        "reference_genome": "GRCh38",
        "coordinate_system": "BED_0_based_half_open",
        "note": "Fill path for all 47 external concepts; optional sha256 pins each input.",
        "concepts": {
            name: {"path": None, "sha256": None, "source": SOURCES[name]}
            for name in EXTERNAL_CONCEPTS
        },
    }


def _load_sources(path):
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("reference_genome") != "GRCh38":
        raise ValueError("Paper benchmark sources must use GRCh38")
    entries = manifest.get("concepts")
    if not isinstance(entries, dict):
        raise ValueError("Source manifest must contain a concepts object")
    unknown = sorted(set(entries) - set(EXTERNAL_CONCEPTS))
    if unknown:
        raise ValueError(f"Unknown paper benchmark concepts: {unknown}")
    return manifest_path, manifest, entries


def _atomic_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _align_chromosome_names(intervals, annotation_chromosomes):
    """Match paper BED naming (usually `1`) to project naming (`chr1`)."""
    annotation_chromosomes = set(map(str, annotation_chromosomes))
    if not annotation_chromosomes:
        raise ValueError("Annotation matrix has no chromosomes")
    uses_chr = all(chrom.startswith("chr") for chrom in annotation_chromosomes)
    if not uses_chr and any(chrom.startswith("chr") for chrom in annotation_chromosomes):
        raise ValueError("Annotation matrix mixes prefixed and unprefixed chromosome names")
    aligned = {}
    for chrom, values in intervals.items():
        bare = chrom[3:] if chrom.startswith("chr") else chrom
        if bare == "M":
            bare = "MT"
        target = ("chrM" if bare == "MT" else f"chr{bare}") if uses_chr else bare
        if target in annotation_chromosomes:
            aligned[target] = values
    return aligned


def build_paper_matrix(ann_path, source_path, out_ann, out_panel, out_prevalence):
    """Build an isolated 55-concept matrix; publish outputs only after validation."""
    ann = read_annotation(ann_path)
    manifest_path, manifest, entries = _load_sources(source_path)
    missing_base = sorted(set(SHARED_COLUMNS.values()) - set(ann.columns))
    if missing_base:
        raise ValueError(f"Base matrix is missing shared cCRE concepts: {missing_base}")

    missing_entries = sorted(set(EXTERNAL_CONCEPTS) - set(entries))
    unresolved = []
    resolved = {}
    for name in EXTERNAL_CONCEPTS:
        entry = entries.get(name, {})
        raw_path = entry.get("path") if isinstance(entry, dict) else None
        if not raw_path:
            unresolved.append(name)
            continue
        bed_path = Path(raw_path)
        if not bed_path.is_absolute():
            bed_path = manifest_path.parent / bed_path
        if not bed_path.is_file():
            unresolved.append(name)
            continue
        expected = entry.get("sha256")
        actual = sha256(bed_path)
        if expected and expected != actual:
            raise ValueError(f"SHA-256 mismatch for {name}: {bed_path}")
        resolved[name] = (bed_path, actual, entry.get("source", SOURCES[name]))
    if missing_entries or unresolved:
        absent = sorted(set(missing_entries + unresolved))
        raise ValueError(f"Paper benchmark requires all 47 external BED inputs; missing: {absent}")

    meta = ["chrom", "bin_start", "bin_end", "split", "n_mask"]
    if "window_idx" in ann:
        meta.append("window_idx")
    output = ann[meta].copy()
    provenance = {}
    for paper_name, project_name in SHARED_COLUMNS.items():
        output[paper_name] = ann[project_name].astype(np.uint8)
        provenance[paper_name] = {
            "source": SOURCES[paper_name], "reused_column": project_name,
            "base_matrix": str(Path(ann_path).resolve()),
        }
    for name in EXTERNAL_CONCEPTS:
        bed_path, digest, declared_source = resolved[name]
        intervals = _align_chromosome_names(read_bed(str(bed_path)), output.chrom.unique())
        output[name] = mark_bins(output, intervals)
        provenance[name] = {
            "source": declared_source,
            "path": str(bed_path.resolve()),
            "sha256": digest,
            "n_intervals": int(sum(len(values) for values in intervals.values())),
        }

    valid = output.n_mask.to_numpy(dtype=bool)
    rows = []
    for name in CONCEPTS:
        rows.append({
            "concept": name,
            "group": next(group for group, names in GROUPS.items() if name in names),
            "prevalence_valid_bins": float(output.loc[valid, name].mean()),
            "n_positive_valid_bins": int(output.loc[valid, name].sum()),
            "sparse_event": name in SPARSE_EVENT_CONCEPTS,
            "source": SOURCES[name],
        })
    prevalence = pd.DataFrame(rows)
    constant = prevalence.loc[
        prevalence.n_positive_valid_bins.isin([0, int(valid.sum())]), "concept"
    ].tolist()
    if constant:
        raise ValueError(f"Benchmark concepts are constant on valid bins: {constant}")

    panel = {
        "schema_version": 1,
        "benchmark": "nair_icml2026_55_concepts",
        "analysis_role": "external_benchmark_only",
        "paper_url": PAPER_URL,
        "repository_url": REPO_URL,
        "data_doi": DOI,
        "reference_genome": "GRCh38",
        "bin_bp": BIN_BP,
        "confirmatory": [],
        "exploratory": [],
        "negative_control": [],
        "artifact_control": [],
        "benchmark_only": CONCEPTS,
        "groups": GROUPS,
        "sparse_event": SPARSE_EVENT_CONCEPTS,
        "event_enrichment": EVENT_ENRICHMENT,
        "directional_in_source_paper": DIRECTIONAL_CONCEPTS,
        "orientation_handling": "strand-collapsed overlap labels on the 128-bp reference grid",
        "shared_with_primary": SHARED_COLUMNS,
        "external_source_concepts": EXTERNAL_CONCEPTS,
        "source_manifest": str(manifest_path),
        "provenance": provenance,
        "separation_note": (
            "This benchmark does not alter concept_panel.json, its FDR family, "
            "or the pre-registered AlphaGenome intervention readout map."
        ),
    }

    _atomic_parquet(output, out_ann)
    _atomic_parquet(prevalence, out_prevalence)
    atomic_json(out_panel, panel)
    return output, panel, prevalence


def cmd_paper(argv):
    parser = argparse.ArgumentParser(description="Build the isolated ICML 2026 paper benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="write the 47-source manifest template")
    template.add_argument("--out", default="paper_benchmark_sources.json")
    matrix = sub.add_parser("matrix", help="build a separate 55-concept annotation matrix")
    matrix.add_argument("--ann", required=True, help="existing matrix containing the 8 shared cCRE columns")
    matrix.add_argument("--sources", required=True, help="completed template with 47 BED paths")
    matrix.add_argument("--out-ann", default="paper_benchmark_matrix.parquet")
    matrix.add_argument("--out-panel", default="paper_benchmark_panel.json")
    matrix.add_argument("--out-prevalence", default="paper_benchmark_prevalence.parquet")
    args = parser.parse_args(argv)

    if args.command == "template":
        atomic_json(args.out, source_template())
        print(f"wrote {args.out}: 47 external sources; 8 cCRE concepts are reused")
        return 0
    build_paper_matrix(args.ann, args.sources, args.out_ann, args.out_panel,
                       args.out_prevalence)
    print(f"wrote isolated paper benchmark: {args.out_ann} (55 concepts)")
    return 0
