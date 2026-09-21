import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ag_sae import match as matching
from ag_sae import paper
from ag_sae.statistics import (average_precision, domain_precision_recall_f1,
                               event_enrichment_with_block_bootstrap,
                               genomic_block_ids)
from tests.cache_fixture import write_cache


def base_matrix(path):
    ann = pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "bin_start": [0, 128, 256, 384],
        "bin_end": [128, 256, 384, 512],
        "split": ["test"] * 4,
        "n_mask": [True] * 4,
        "unrelated_primary_column": [0, 1, 0, 1],
    })
    for i, project_name in enumerate(paper.SHARED_COLUMNS.values()):
        ann[project_name] = [i % 2, 1 - i % 2, i % 2, 1 - i % 2]
    ann.to_parquet(path, index=False)
    return ann


def completed_sources(tmp_path):
    bed = tmp_path / "shared_external_source.bed"
    # The paper's processing scripts strip the `chr` prefix; the project matrix
    # keeps it. The benchmark builder must align these conventions explicitly.
    bed.write_text("1\t0\t128\n1\t256\t384\n")
    manifest = paper.source_template()
    for entry in manifest["concepts"].values():
        entry["path"] = bed.name
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(manifest))
    return path


def test_paper_spec_is_complete_and_separate():
    assert len(paper.CONCEPTS) == 55
    assert len(set(paper.CONCEPTS)) == 55
    assert len(paper.SHARED_COLUMNS) == 8
    assert len(paper.EXTERNAL_CONCEPTS) == 47
    assert not set(paper.EXTERNAL_CONCEPTS) & set(paper.SHARED_COLUMNS)


def test_paper_matrix_builds_exact_external_benchmark(tmp_path):
    ann_path = tmp_path / "base.parquet"
    base_matrix(ann_path)
    sources = completed_sources(tmp_path)
    out_ann = tmp_path / "paper.parquet"
    out_panel = tmp_path / "paper.json"
    out_prev = tmp_path / "prevalence.parquet"
    matrix, panel, prevalence = paper.build_paper_matrix(
        ann_path, sources, out_ann, out_panel, out_prev)

    meta = {"chrom", "bin_start", "bin_end", "split", "n_mask"}
    assert set(matrix) - meta == set(paper.CONCEPTS)
    assert "unrelated_primary_column" not in matrix
    assert panel["benchmark_only"] == paper.CONCEPTS
    assert panel["confirmatory"] == []
    assert len(prevalence) == 55
    concepts, primary, exploratory, roles = matching.resolve_concepts(matrix, out_panel)
    assert concepts == paper.CONCEPTS
    assert primary == []
    assert exploratory == paper.CONCEPTS
    assert roles["benchmark_only"] == paper.CONCEPTS


def test_paper_matrix_refuses_partial_sources_without_outputs(tmp_path):
    ann_path = tmp_path / "base.parquet"
    base_matrix(ann_path)
    manifest = paper.source_template()
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(manifest))
    outputs = [tmp_path / "paper.parquet", tmp_path / "paper.json", tmp_path / "prev.parquet"]
    with pytest.raises(ValueError, match="all 47"):
        paper.build_paper_matrix(ann_path, source_path, *outputs)
    assert not any(path.exists() for path in outputs)


def test_match_accepts_benchmark_only_panel_without_primary_claims(tmp_path):
    ann = pd.DataFrame({
        "chrom": ["chr1"] * 16,
        "bin_start": np.arange(16) * 128,
        "bin_end": (np.arange(16) + 1) * 128,
        "split": ["test"] * 16,
        "n_mask": [True] * 16,
        "paper_a": [0] * 8 + [1] * 8,
        "paper_b": [0, 1] * 8,
        "must_not_be_scored": [1, 0, 0, 0] * 4,
    })
    ann_path = tmp_path / "ann.parquet"
    ann.to_parquet(ann_path, index=False)
    write_cache(tmp_path, ann, np.column_stack([np.arange(16), np.arange(16) % 3]).astype(np.float32))
    panel = {
        "confirmatory": [], "exploratory": [], "negative_control": [],
        "artifact_control": [], "benchmark_only": ["paper_a", "paper_b"],
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel))
    out = tmp_path / "out"
    matching.cmd_match([
        "--mode", "raw", "--act-dir", str(tmp_path), "--ann", str(ann_path),
        "--panel", str(panel_path), "--out", str(out), "--device", "cpu",
        "--n-control", "9",
    ])
    result = pd.read_csv(out / "match_raw_test.csv")
    assert result.concept.tolist() == ["paper_a", "paper_b"]
    assert set(result.panel) == {"benchmark_only"}
    summary = json.loads((out / "summary_raw_test.json").read_text())
    assert summary["primary_concepts"] == []
    assert summary["benchmark_concepts"] == ["paper_a", "paper_b"]
    assert summary["fdr_family"] == "external_benchmark"
    assert summary["fdr_family_concepts"] == ["paper_a", "paper_b"]
    assert summary["mean_best_auroc_primary"] is None


def test_average_precision_uses_prevalence_baseline_and_ties():
    labels = np.array([1, 0, 1, 0], dtype=bool)
    assert average_precision(labels, np.ones(4)) == pytest.approx(0.5)
    assert average_precision(labels, [4, 1, 3, 2]) == pytest.approx(1.0)
    assert average_precision(labels, [1, 4, 2, 3]) == pytest.approx(0.4166666667)


def test_domain_metric_breaks_at_coordinate_gaps_and_chromosomes():
    labels = [1, 1, 1, 1, 0]
    predicted = [0, 1, 0, 1, 1]
    result = domain_precision_recall_f1(
        predicted, labels,
        ["chr1", "chr1", "chr1", "chr2", "chr2"],
        [0, 128, 512, 0, 128])
    assert result["n_domains"] == 3
    assert result["n_domains_recovered"] == 2
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


def test_sparse_event_enrichment_and_block_bootstrap_are_window_based():
    # Four identical 1-kb blocks. Each contains one event with activation 2 in
    # its +/-1-bin neighbourhood; the remaining background activation is 1.
    n_blocks, bins_per_block = 4, 8
    starts = np.arange(n_blocks * bins_per_block) * 128
    chrom = np.array(["chr1"] * len(starts))
    labels = np.zeros(len(starts), dtype=bool)
    scores = np.ones(len(starts), dtype=float)
    for block in range(n_blocks):
        center = block * bins_per_block + 3
        labels[center] = True
        scores[center - 1:center + 2] = 2.0
    blocks = genomic_block_ids(chrom, starts, block_bp=bins_per_block * 128)
    result = event_enrichment_with_block_bootstrap(
        scores, labels, chrom, starts, blocks, radius_bins=1, sigma_bins=1.0,
        n_bootstrap=200, seed=17)
    assert result["n_events"] == 4
    assert result["n_blocks"] == 4
    assert result["event_mean"] == pytest.approx(2.0)
    assert result["background_mean"] == pytest.approx(1.0)
    assert result["log_enrichment"] == pytest.approx(np.log(2))
    assert result["ci_low"] == pytest.approx(np.log(2))
    assert result["ci_high"] == pytest.approx(np.log(2))
    assert result["bootstrap_valid"] == 200


def test_event_windows_do_not_cross_coordinate_gaps():
    scores = np.array([1.0, 4.0, 100.0, 1.0])
    labels = np.array([0, 1, 0, 0], dtype=bool)
    chrom = np.array(["chr1"] * 4)
    starts = np.array([0, 128, 1024, 1152])
    blocks = genomic_block_ids(chrom, starts, block_bp=512)
    result = event_enrichment_with_block_bootstrap(
        scores, labels, chrom, starts, blocks, radius_bins=1, sigma_bins=1.0,
        n_bootstrap=50, seed=1)
    # The very large score after the coordinate gap is background, never part
    # of the event's Gaussian neighbourhood.
    expected_event = (np.exp(-0.5) * 1.0 + 4.0) / (np.exp(-0.5) + 1.0)
    assert result["event_mean"] == pytest.approx(expected_event)
    assert result["background_mean"] == pytest.approx(50.5)
