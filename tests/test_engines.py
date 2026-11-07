"""The ported intervention engines, end to end on the fake AlphaGenome."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from ag_sae import intervene as I
from ag_sae import motif_intervene as M
from ag_sae.edit import capture_residual
from ag_sae.sae import BorzoiSAE

from tests.fake_alphagenome import FakeAlphaGenome

BINS = 32
L = BINS * 128
DIM = 16


class _Tap:
    kind, key, bin_bp, name = "tower", 2, 128, "resid_pre_b2"


@pytest.fixture(scope="module")
def model():
    return FakeAlphaGenome(dim=DIM, n_blocks=4).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def sae():
    torch.manual_seed(3)
    return BorzoiSAE(DIM, 64, 6, torch.linspace(1, 2, DIM)).eval().requires_grad_(False)


@pytest.fixture()
def fasta(tmp_path):
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 3 * L))
    path = tmp_path / "genome.fa"
    lines = [seq[i:i + 60] for i in range(0, len(seq), 60)]
    path.write_text(">chr1 test\n" + "\n".join(lines) + "\n")
    return path, seq


class TestWindowSequence:
    def test_seeks_the_same_bases_as_a_full_read(self, fasta):
        path, seq = fasta
        assert I.window_sequence(path, "chr1", 1000, 1500) == seq[1000:1500]
        assert I.window_sequence(path, "1", 0, 10) == seq[:10]          # chr-less alias


class TestTriad:
    def annotations(self, start):
        starts = start + np.arange(BINS) * 128
        frame = pd.DataFrame({"chrom": "chr1", "bin_start": starts, "split": "test"})
        for col, bins in (("cCRE_PLS", [3, 4]), ("cCRE_CTCF_only", [10]), ("cCRE_pELS", [20, 21]),
                          ("shuffled_dELS", [25])):
            frame[col] = np.isin(np.arange(BINS), bins)
        frame["cCRE_any"] = frame[["cCRE_PLS", "cCRE_CTCF_only", "cCRE_pELS"]].any(axis=1)
        return frame

    def test_batching_does_not_change_the_results(self, model, sae, fasta, tmp_path):
        path, _ = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}, {"chrom": "chr1", "start": L, "end": 2 * L}]
        ann = pd.concat([self.annotations(w["start"]) for w in windows], ignore_index=True)
        feats = I.DiscoveredFeatures(1, 2, 3, 4)
        runs = {}
        for max_batch in (1, 6):
            results, summary = I.run_triad_ablation(sae, model, _Tap, windows, ann, feats, path,
                                                    torch.tensor([0]), tmp_path / f"b{max_batch}",
                                                    max_batch=max_batch)
            runs[max_batch] = results
        assert json.dumps(runs[1], sort_keys=True) == json.dumps(runs[6], sort_keys=True)
        assert (tmp_path / "b6" / "regulatory_triad_summary.json").is_file()

    def test_discovery_reads_the_stream_before_the_tap(self, model, sae, fasta):
        path, seq = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}]
        found = I.discover_concept_features(sae, model, _Tap, windows, self.annotations(0), path,
                                            torch.tensor([0]))
        x = torch.from_numpy(I.sequence_to_onehot(seq[:L]).astype(np.float32))[None]
        codes, _ = sae.encode_codes(capture_residual(model, 2, x)[0])
        pos = np.isin(np.arange(BINS), [10])
        expected = int(np.argmax((codes[pos].mean(0) - codes[~pos].mean(0)).numpy()))
        assert found.insulator_feature_id == expected


class TestMatchedControl:
    def test_control_change_has_the_target_change_l2(self, sae):
        torch.manual_seed(0)
        rows = torch.randn(1, DIM) * 3
        memo = {}
        out = I.matched_control_edit(sae, 0, 1, 5.0, 3.5)(rows, memo)
        codes, params = sae.encode_codes(rows)
        base = sae.decode_codes(codes, params)
        target = codes.clone()
        target[:, 0] = torch.clamp_min(codes[:, 0], 5.0)
        target_l2 = (sae.decode_codes(target, params) - base).norm()
        assert "unmeasurable" not in memo
        torch.testing.assert_close((out - base).norm(), target_l2, rtol=1e-4, atol=1e-5)

    def test_flags_a_zero_change(self, sae):
        rows = torch.randn(1, DIM)
        memo = {}
        I.matched_control_edit(sae, 0, 1, 0.0, 0.0)(rows, memo)   # clamping at 0 changes nothing
        assert "unmeasurable" in memo


class TestMotifSummary:
    def test_ratio_of_means_alongside_clipped_per_window_ratios(self):
        motif = M.get_canonical_ctcf_motif(None)
        rows = []
        for ism, abl in ((-1.0, -0.5), (-0.00001, -0.4), (-2.0, -0.2)):
            heads = {h: {"delta_ism": ism, "delta_abl": abl, "delta_rescue": 0.1,
                         "mediation_ratio": M._ratio(abl, ism), "rescue_ratio": 0.0}
                     for h in M.MOTIF_HEADS}
            rows.append({"heads": heads})
        summary = M.summarize_motif_windows(rows, motif, 7)
        tf = summary["heads"]["chip_tf"]
        assert tf["mediation_ratio_of_means"] == pytest.approx((-1.1 / 3) / (-3.00001 / 3))
        # The tiny-ISM window contributes 0 to the per-window mean.
        assert tf["mean_mediation_ratio"] == pytest.approx((0.5 + 0.0 + 0.1) / 3)

    def test_subdir_naming(self):
        assert M.matrix_subdir("sae") == "motif_matrix"
        assert M.matrix_subdir("sae_k64") == "motif_matrix_k64"
        assert M.matrix_subdir("sae_v2") == "motif_matrix_v2"


class TestMotifCohort:
    def test_prepare_and_measure_end_to_end(self, model, sae, tmp_path):
        motif = M.get_canonical_ctcf_motif(None)
        rng = np.random.default_rng(1)
        seq = list("".join(rng.choice(list("ACGT"), 2 * L)))
        for at in (700, L + 2000):                       # one strong site per window
            seq[at:at + motif.length] = motif.consensus
        seq = "".join(seq)
        path = tmp_path / "g.fa"
        path.write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
        windows = [{"chrom": "chr1", "start": 0, "end": L}, {"chrom": "chr1", "start": L, "end": 2 * L}]

        prepared = M.prepare_motif_windows(windows, path, motif, n_windows=5, p_threshold=1e-5,
                                           log=lambda *a: None)
        assert [w.site_start for w in prepared] == [700, 2000]
        again = M.prepare_motif_windows(windows, path, motif, n_windows=5, p_threshold=1e-5,
                                        log=lambda *a: None)
        assert all(torch.equal(a.mut, b.mut) for a, b in zip(prepared, again))   # reproducible ISM

        summary = M.run_motif_causal_cohort(sae, model, _Tap, prepared, motif, 3, tmp_path / "out",
                                            max_batch=2, device="cpu")
        assert summary["n_measured_windows"] == 2
        per_window = json.loads((tmp_path / "out" / "motif_per_window_results.json").read_text())
        assert per_window[0]["anchor_bin"] == 700 // 128
        assert set(per_window[0]["heads"]) == set(M.MOTIF_HEADS)
