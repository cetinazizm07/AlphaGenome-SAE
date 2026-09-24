"""Coordinate and selection checks for the precommitted gene TSS pilot."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from gene_tss_readout import extract_gene_readouts, match_control_norm, summarize_gene_effects
from freeze_gene_tss_cohort import freeze
from plan_fullstream_cohort import sha256
from run_gene_tss_cohort import run as run_cohort
from plan_gene_tss_injection import (
    BIN_BP,
    WINDOW_BP,
    candidate_at_rank,
    eligible_genes,
    exon_bin_weights,
    gtf_attributes,
    read_mane_transcripts,
)


def gtf_record(feature: str, start1: int, end1: int, strand: str,
               gene: str, transcript: str) -> str:
    attrs = (f'gene_id "{gene}"; transcript_id "{transcript}"; '
             f'gene_type "protein_coding"; gene_name "{gene}"; '
             'transcript_type "protein_coding"; tag "basic"; tag "MANE_Select";')
    return f"chr1\tHAVANA\t{feature}\t{start1}\t{end1}\t.\t{strand}\t.\t{attrs}\n"


class TssPlanTests(unittest.TestCase):
    def test_repeated_tags_and_minus_strand_tss(self):
        attrs = gtf_attributes('tag "basic"; tag "MANE_Select"; gene_id "ENSG1";')
        self.assertEqual(attrs["tag"], ["basic", "MANE_Select"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genes.gtf"
            path.write_text(gtf_record("transcript", 200_001, 220_000, "+", "A", "AT")
                            + gtf_record("transcript", 390_001, 410_000, "-", "B", "BT"))
            records = read_mane_transcripts(path, {"chr1"})
        by_gene = {record["gene_id"]: record for record in records}
        self.assertEqual(by_gene["A"]["tss0"], 200_000)
        self.assertEqual(by_gene["B"]["tss0"], 409_999)

    def test_exon_overlap_is_counted_once(self):
        weights = exon_bin_weights([(0, 100), (50, 150)], 0)
        self.assertEqual(weights, [{"bin_index": 0, "exon_bp": 128},
                                   {"bin_index": 1, "exon_bp": 22}])

    def test_rank_selection_is_fixed_and_bounded(self):
        candidates = [{"gene_id": "A"}, {"gene_id": "B"}]
        self.assertEqual(candidate_at_rank(candidates, 1), {"gene_id": "B"})
        with self.assertRaises(ValueError):
            candidate_at_rank(candidates, -1)
        with self.assertRaises(ValueError):
            candidate_at_rank(candidates, 2)

    def test_selection_uses_annotation_and_exons_without_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genes.gtf"
            path.write_text(
                gtf_record("transcript", 200_001, 220_000, "+", "A", "AT")
                + gtf_record("exon", 200_001, 203_000, "+", "A", "AT")
                + gtf_record("transcript", 390_001, 410_000, "-", "B", "BT")
                + gtf_record("exon", 407_001, 410_000, "-", "B", "BT")
            )
            starts = BIN_BP * np.arange(WINDOW_BP // BIN_BP)
            annotations = pd.DataFrame({
                "split": "test", "chrom": "chr1", "bin_start": starts,
                "n_mask": True, "cCRE_PLS": False,
            })
            manifest = pd.DataFrame([{
                "split": "test", "chrom": "chr1",
                "win_start": 0, "win_end": WINDOW_BP,
            }])
            selected = eligible_genes(path, manifest, annotations)
            self.assertEqual({gene["gene_id"] for gene in selected}, {"A", "B"})
            self.assertEqual(selected, eligible_genes(path, manifest, annotations))
            annotations.loc[annotations.bin_start == 200_000 // BIN_BP * BIN_BP,
                            "cCRE_PLS"] = True
            selected_after = eligible_genes(path, manifest, annotations)
            self.assertEqual([gene["gene_id"] for gene in selected_after], ["B"])

    def test_gene_exon_readout_and_tss_control_dose(self):
        gene = {"tss_bin_index": 4,
                "exon_bin_weights": [{"bin_index": 2, "exon_bp": 32},
                                       {"bin_index": 3, "exon_bp": 96}]}
        tracks = {head: [{"index": 0}, {"index": 2}]
                  for head in ("rna_seq", "cage", "procap")}
        rna = torch.zeros((1, 8192, 3))
        rna[0, 2, [0, 2]] = torch.tensor([2., 4.])
        rna[0, 3, [0, 2]] = torch.tensor([6., 8.])
        tss = torch.zeros((1, 8192, 3))
        tss[0, 4, [0, 2]] = torch.tensor([3., 5.])
        output = {"rna_seq": {128: rna}, "cage": {128: tss}, "procap": {128: tss}}
        values = extract_gene_readouts(output, gene, tracks)
        np.testing.assert_allclose(values["rna_seq"], [5., 7.])
        np.testing.assert_allclose(values["cage"], [3., 5.])
        readings = {f"{head}_{label}": values[head] * scale
                    for head in tracks for label, scale in
                    (("original", 1.), ("reconstruction", 1.),
                     ("target", 2.), ("control", 1.))}
        summary = summarize_gene_effects(readings, tracks)
        self.assertGreater(summary["rna_seq"]["target_median_log2_ratio"], 0)
        self.assertEqual(summary["rna_seq"]["control_median_log2_ratio"], 0)
        self.assertEqual(summary["rna_seq"]["highest_original_expression_track"]["track_index"], 2)

        baseline = torch.zeros((1, 8, 2))
        target, control = baseline.clone(), baseline.clone()
        target[0, 4, 0] = 2.
        control[0, 4, 1] = 1.
        matched, norms = match_control_norm(baseline, target, control, 4)
        torch.testing.assert_close(matched[0, 4], torch.tensor([0., 2.]))
        torch.testing.assert_close(matched[0, :4], baseline[0, :4])
        self.assertEqual(norms["control_scale"], 2.0)

    def test_primary_rna_track_ignores_intervention_outcome(self):
        tracks = {head: [{"index": 7, "biosample": "A"},
                         {"index": 9, "biosample": "B"}]
                  for head in ("rna_seq", "cage", "procap")}
        readings = {f"{head}_{label}": np.array(values, dtype=np.float32)
                    for head in tracks for label, values in (
                        ("original", [10.0, 1.0]),
                        ("reconstruction", [9.0, 1.0]),
                        ("target", [8.0, 100.0]),
                        ("control", [9.0, 1.0]),
                    )}
        primary = summarize_gene_effects(readings, tracks)["rna_seq"][
            "highest_original_expression_track"
        ]
        self.assertEqual(primary["track_index"], 7)
        self.assertEqual(primary["biosample"], "A")
        self.assertLess(primary["target_log2_ratio"], 0)

    def test_freeze_cohort_before_future_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.json"
            gtf = root / "genes.gtf"
            pilot = root / "pilot.json"
            receipt = root / "receipt.json"
            parent.write_text('{"source_sha256": {}}')
            gtf.write_text("fixed-gtf")
            candidates = [{"gene_id": f"G{i}", "gene_name": f"Gene{i}",
                           "transcript_id": f"T{i}"} for i in range(3)]
            pilot.write_text(json.dumps({
                "gene": candidates[0], "parent_plan_sha256": sha256(parent),
                "gtf_sha256": sha256(gtf),
            }))
            receipt.write_text(json.dumps({
                "status": "measured", "plan_sha256": sha256(pilot),
                "readouts_sha256": "fixed-readout-hash",
            }))

            def plan_for_rank(_parent_path, _gtf_path, _parent, _candidates, rank):
                return {"gene": candidates[rank]}

            with patch("freeze_gene_tss_cohort.validated_context",
                       return_value=({"source_sha256": {}}, candidates)), patch(
                           "freeze_gene_tss_cohort.plan_at_rank", side_effect=plan_for_rank):
                result = freeze(parent, gtf, pilot, receipt, root / "cohort", 3)
            self.assertEqual(result["genes"], ["Gene0", "Gene1", "Gene2"])
            manifest = json.loads((root / "cohort" / "cohort_manifest.json").read_text())
            self.assertEqual([row["status"] for row in manifest["records"]],
                             ["existing_technical_pilot", "planned", "planned"])
            with self.assertRaises(ValueError):
                freeze(parent, gtf, pilot, receipt, root / "cohort", 3)

    def test_cohort_manager_reuses_only_hash_validated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort"
            cohort.mkdir()
            plan = cohort / "rank01_plan.json"
            plan.write_text('{"gene": "G1"}')
            manifest = {
                "format": "exploratory_fold1_b8_gene_tss_cohort_v1",
                "fold": "fold1", "tap": "resid_pre_b8", "sae_seed": 0,
                "rank_cutoff_exclusive": 2,
                "records": [
                    {"rank": 0, "status": "existing_technical_pilot"},
                    {"rank": 1, "gene_id": "G1", "gene_name": "Gene1",
                     "plan_file": plan.name, "plan_sha256": sha256(plan)},
                ],
            }
            (cohort / "cohort_manifest.json").write_text(json.dumps(manifest))
            output = root / "outputs" / "rank01"
            output.mkdir(parents=True)
            (output / "plan.json").write_bytes(plan.read_bytes())
            (output / "readouts.npz").write_bytes(b"fixed-readouts")
            receipt = {
                "format": "fullstream_gene_tss_injection_pilot_v1",
                "plan_sha256": sha256(plan), "gene": {"gene_id": "G1"},
                "status": "measured",
                "readouts_sha256": sha256(output / "readouts.npz"),
            }
            (output / "receipt.json").write_text(json.dumps(receipt))
            worker = Path(__file__).with_name("run_gene_tss_injection_pilot.py")
            result = run_cohort(cohort, root / "outputs", root, root, worker)
            self.assertEqual(result["completed"], 1)
            self.assertIsNone(json.loads((root / "outputs" /
                                          "campaign_status.json").read_text())["records"][0][
                                              "elapsed_wall_seconds"])
            (output / "readouts.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                run_cohort(cohort, root / "outputs", root, root, worker)


if __name__ == "__main__":
    unittest.main()
