# External 55-concept benchmark

The ICML 2026 benchmark from Nair et al. is intentionally kept separate from
the frozen project panel in `concept_panel.json`. It is an external
replication/coverage test; it does not add confirmatory hypotheses and it does
not change the AlphaGenome intervention readout map.

## Build the source manifest

```bash
python -m ag_sae paper template --out paper_benchmark_sources.json
```

Fill all 47 `path` fields with GRCh38, BED-format annotations. Relative paths
are resolved relative to the manifest. Add `sha256` values to pin downloaded
files. The remaining eight ENCODE Registry V4 cCRE columns are reused from the
existing project matrix.

## Build one isolated matrix per fold

```bash
python -m ag_sae paper matrix \
  --ann annotation_matrix_fold0.parquet \
  --sources paper_benchmark_sources.json \
  --out-ann paper_benchmark_matrix_fold0.parquet \
  --out-panel paper_benchmark_panel_fold0.json \
  --out-prevalence paper_benchmark_prevalence_fold0.parquet
```

The command refuses partial input, wrong reference-genome declarations,
constant labels, unknown concepts, and checksum mismatches. Its output contains
exactly the 55 paper concepts plus coordinate/mask columns.

The source paper reverse-complements negative-strand examples for directional
annotations. This adaptation operates on AlphaGenome's existing 128-bp
reference grid and therefore collapses strand into one overlap label. The panel
records every affected concept under `directional_in_source_paper` and records
the adaptation in `orientation_handling`; results must be described as a
paper-derived benchmark, not a bit-for-bit replication.

Run matching with the resulting panel:

```bash
python -m ag_sae match \
  --mode sae \
  --sae SAE_CHECKPOINT.pt \
  --act-dir ACTIVATION_CACHE \
  --ann paper_benchmark_matrix_fold0.parquet \
  --panel paper_benchmark_panel_fold0.json \
  --split test \
  --n-control 1199 \
  --out paper_benchmark_results/fold0
```

`1199` permutations make the smallest empirical p-value `1/1200`, fine enough
to reach the rank-one Benjamini-Hochberg threshold for 55 tests at q=0.05.
The paper concepts are reported as `benchmark_only`: they do not enter the
primary panel's FDR family or intervention stage.

## Complementary statistics

The match table now also reports:

- `best_feature_auprc`: average precision of the AUROC-selected feature;
- `auprc_lift`: AUPRC divided by positive prevalence;
- `rank_biserial`: effect-size form `2 * AUROC - 1`;
- `domain_precision`, `domain_recall`, and `domain_f1`: the paper's positive
  activation (`> 0`) domain metric for positive SAE matches.

AUPRC and domain F1 are descriptive companion values for the selected feature.
The empirical p-value remains based on the selection-aware, maximum-feature
AUROC permutation null.

Sparse events (`MANE_TSS`, start/stop codons, `miRNA`, and `selenocysteine`)
also receive the precommitted resolution-aware event analysis stored in the
panel under `event_enrichment`:

- Gaussian event radius: 2 bins (256 bp);
- Gaussian sigma: 1 bin (128 bp);
- bootstrap unit: one 131,072-bp AlphaGenome window;
- bootstrap replicates: 2,000 with seed 20260921;
- confidence interval: percentile 95%.

The result columns are `event_log_enrichment`, `event_enrichment_ci_low`,
`event_enrichment_ci_high`, `event_activation_mean`,
`event_background_mean`, `event_n`, `event_background_n`, and
`event_bootstrap_valid`. Event enrichment is reported only for positive TopK
SAE matches: raw neurons and inverse matches are left missing because the
nonnegative log-ratio interpretation does not apply to them. AlphaGenome's SAE
grid is 128 bp, so the paper's exact 10-bp kernel is deliberately not copied.
The bootstrap interval is conditional on the AUROC-selected feature; the
separate maximum-feature circular-permutation test continues to account for
feature selection and provides the inferential p-value.
