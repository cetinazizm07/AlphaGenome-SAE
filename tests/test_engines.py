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


def assert_close_tree(a, b, rel=1e-5, abs_=1e-6):
    """Equal structure and values, floats up to batch-size rounding."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            assert_close_tree(a[k], b[k], rel, abs_)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_close_tree(x, y, rel, abs_)
    elif isinstance(a, float):
        assert b == pytest.approx(a, rel=rel, abs=abs_)
    else:
        assert a == b


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
        assert_close_tree(runs[1], runs[6])
        assert (tmp_path / "b6" / "regulatory_triad_summary.json").is_file()

    def test_discovery_reads_the_stream_before_the_tap(self, model, sae, fasta):
        path, seq = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}]
        found = I.discover_concept_features(sae, model, _Tap, windows, self.annotations(0), path,
                                            torch.tensor([0]), score="mean_diff")
        x = torch.from_numpy(I.sequence_to_onehot(seq[:L]).astype(np.float32))[None]
        codes, _ = sae.encode_codes(capture_residual(model, 2, x)[0])
        pos = np.isin(np.arange(BINS), [10])
        expected = int(np.argmax((codes[pos].mean(0) - codes[~pos].mean(0)).numpy()))
        assert found.insulator_feature_id == expected

    def test_discovery_ranks_by_cohens_d(self, model, sae, fasta):
        path, seq = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}]
        found = I.discover_concept_features(sae, model, _Tap, windows, self.annotations(0), path,
                                            torch.tensor([0]), top_n=3, min_fire_rate=0.0)
        x = torch.from_numpy(I.sequence_to_onehot(seq[:L]).astype(np.float32))[None]
        codes = sae.encode_codes(capture_residual(model, 2, x)[0])[0].double().numpy()
        pos = np.isin(np.arange(BINS), [20, 21])
        p, q = codes[pos], codes[~pos]
        pooled = np.sqrt((((p - p.mean(0)) ** 2).sum(0) + ((q - q.mean(0)) ** 2).sum(0)) / (BINS - 2))
        d = np.where(pooled > 1e-12, (p.mean(0) - q.mean(0)) / np.where(pooled > 0, pooled, 1), 0.0)
        assert found.ranked["cCRE_pELS"] == tuple(int(i) for i in np.argsort(-d, kind="stable")[:3])
        assert found.enhancer_feature_id == found.ranked["cCRE_pELS"][0]
        np.testing.assert_allclose(found.scores["cCRE_pELS"], np.sort(d)[::-1][:3], rtol=1e-3)  # float32 means
        assert found.top("cCRE_pELS", 2) == found.ranked["cCRE_pELS"][:2]

    def test_triad_reports_both_methods_and_a_joint_ablation(self, model, sae, fasta, tmp_path):
        path, _ = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}, {"chrom": "chr1", "start": L, "end": 2 * L}]
        ann = pd.concat([self.annotations(w["start"]) for w in windows], ignore_index=True)
        ranked = {"cCRE_PLS": (1, 5, 7), "cCRE_CTCF_only": (2,), "cCRE_pELS": (3, 9),
                  "shuffled_dELS": (4,)}
        feats = I.DiscoveredFeatures(1, 2, 3, 4, ranked=ranked, score="cohen_d")
        runs = {}
        for max_batch in (2, 16):
            results, summary = I.run_triad_ablation(sae, model, _Tap, windows, ann, feats, path,
                                                    torch.tensor([0]), tmp_path / f"b{max_batch}",
                                                    max_batch=max_batch, top_n=3)
            runs[max_batch] = results
        assert json.dumps(runs[2], sort_keys=True) == json.dumps(runs[16], sort_keys=True)
        assert set(summary) == {"PLS_Promoter", "CTCF_Insulator", "Enhancer_pELS", "Shuffled_Control"}
        assert summary["PLS_Promoter"]["joint"]["feature_ids"] == [1, 5, 7]
        assert summary["Enhancer_pELS"]["joint"]["feature_ids"] == [3, 9]
        assert "joint" not in summary["CTCF_Insulator"]
        for entry in summary.values():
            assert set(entry["heads"]) == set(entry["heads_error_preserving"]) == set(I.ALL_HEADS)

    def test_error_preserving_ablation_of_a_silent_feature_changes_nothing(self, model, sae, fasta,
                                                                           tmp_path):
        path, seq = fasta
        window = {"chrom": "chr1", "start": 0, "end": L}
        x = torch.from_numpy(I.sequence_to_onehot(seq[:L]).astype(np.float32))[None]
        codes = sae.encode_codes(capture_residual(model, 2, x)[0])[0]
        silent = int(torch.nonzero(codes.sum(0) == 0)[0])
        feats = I.DiscoveredFeatures(silent, silent, silent, silent)
        results, _ = I.run_triad_ablation(sae, model, _Tap, [window], self.annotations(0), feats, path,
                                          torch.tensor([0]), tmp_path)
        for entry in results[0]["features"].values():
            for head in I.ALL_HEADS:
                # Exactly the original stream; only batch-size rounding remains.
                assert abs(entry["heads_error_preserving"][head]["delta_on_concept"]) < 1e-6


class TestConceptContrast:
    def test_mean_diff_averages_windows_like_the_first_campaign(self):
        rng = np.random.default_rng(0)
        contrast, diffs = I.ConceptContrast(), []
        for n in (10, 30):
            codes = torch.from_numpy(rng.exponential(size=(n, 6)).astype(np.float32))
            pos = np.zeros(n, bool)
            pos[:3] = True
            contrast.add(codes, pos, ~pos)
            diffs.append(codes[:3].mean(0).numpy() - codes[3:].mean(0).numpy())
        np.testing.assert_allclose(contrast.score("mean_diff"), np.mean(diffs, 0), rtol=1e-6)

    def test_rarely_firing_features_are_not_eligible(self):
        codes = torch.zeros(20, 3)
        codes[0, 0] = 100.0                  # fires once, on 1 of 10 positive bins
        codes[:10, 1] = 1.0                  # fires on every positive bin
        pos = np.arange(20) < 10
        contrast = I.ConceptContrast()
        contrast.add(codes, pos, ~pos)
        d = contrast.score("cohen_d", min_fire_rate=0.5)
        assert d[0] == -np.inf and np.isfinite(d[1])
        assert d[2] == -np.inf               # never fires
        assert contrast.score("cohen_d", min_fire_rate=0.0)[2] == 0.0

    def test_cohens_d_contrasts_within_windows(self):
        rng = np.random.default_rng(1)
        contrast = I.ConceptContrast()
        num, n_pos, ss, n_bins = 0.0, 0, 0.0, 0
        for n, shift, k in ((12, 0.0, 3), (40, 10.0, 8), (25, 3.0, 5)):
            codes = rng.exponential(size=(n, 4)) + shift          # window-level offsets
            pos = np.zeros(n, bool)
            pos[:k] = True
            contrast.add(torch.from_numpy(codes.astype(np.float32)), pos, ~pos)
            p, q = codes[pos].astype(np.float32).astype(np.float64), codes[~pos].astype(np.float32).astype(np.float64)
            num = num + k * (p.mean(0) - q.mean(0))
            ss = ss + ((p - p.mean(0)) ** 2).sum(0) + ((q - q.mean(0)) ** 2).sum(0)
            n_pos += k
            n_bins += n
        expected = (num / n_pos) / np.sqrt(ss / (n_bins - 6))
        np.testing.assert_allclose(contrast.score("cohen_d", 0.0), expected, rtol=1e-4)

    def test_empty_contrast_raises(self):
        with pytest.raises(ValueError):
            I.ConceptContrast().score("cohen_d")


class TestFeatureDose:
    def test_quantile_of_nonzero_values(self, model, sae, fasta):
        path, seq = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}, {"chrom": "chr1", "start": L, "end": 2 * L}]
        values = []
        for w in windows:
            x = torch.from_numpy(I.sequence_to_onehot(seq[w["start"]:w["end"]]).astype(np.float32))[None]
            codes = sae.encode_codes(capture_residual(model, 2, x)[0])[0]
            values.append(codes)
        codes = torch.cat(values)
        feature = int(torch.argmax((codes > 0).sum(0)))
        column = codes[:, feature]
        expected = float(np.quantile(column[column > 0].numpy(), 0.9))
        assert I.feature_dose(sae, model, _Tap, windows, path, feature, quantile=0.9) == pytest.approx(expected)

    def test_never_firing_feature_has_no_dose(self, model, sae, fasta):
        path, seq = fasta
        window = {"chrom": "chr1", "start": 0, "end": L}
        x = torch.from_numpy(I.sequence_to_onehot(seq[:L]).astype(np.float32))[None]
        codes = sae.encode_codes(capture_residual(model, 2, x)[0])[0]
        silent = int(torch.nonzero(codes.sum(0) == 0)[0])
        with pytest.raises(ValueError):
            I.feature_dose(sae, model, _Tap, [window], path, silent)


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

    def test_error_preserving_adds_the_same_change_to_the_real_rows(self, sae):
        torch.manual_seed(0)
        rows = torch.randn(1, DIM) * 3
        memo = {}
        on_recon = I.matched_control_edit(sae, 0, 1, 5.0, 3.5)(rows, {})
        on_rows = I.matched_control_edit(sae, 0, 1, 5.0, 3.5, preserve_error=True, flag="u_x1")(rows, memo)
        codes, params = sae.encode_codes(rows)
        base = sae.decode_codes(codes, params)
        torch.testing.assert_close(on_rows - rows, on_recon - base, rtol=1e-4, atol=1e-5)
        assert not memo
        I.matched_control_edit(sae, 0, 1, 0.0, 0.0, preserve_error=True, flag="u_x1")(rows, memo)
        assert "u_x1" in memo


class TestErrorPreservingEdit:
    def test_unchanged_codes_leave_the_rows_untouched(self, sae):
        rows = torch.randn(5, DIM) * 2
        from ag_sae.edit import sae_edit
        assert torch.equal(sae_edit(sae, "r", preserve_error=True)(rows, {}), rows)

    def test_ablation_removes_only_the_feature_contribution(self, sae):
        from ag_sae.edit import sae_edit, zero_feature, zero_features
        torch.manual_seed(1)
        rows = torch.randn(8, DIM) * 2
        codes, params = sae.encode_codes(rows)
        feature = int(torch.argmax((codes > 0).sum(0)))
        ablated = codes.clone()
        ablated[:, feature] = 0
        expected = rows + sae.decode_codes(ablated, params) - sae.decode_codes(codes, params)
        memo = {}
        out = sae_edit(sae, "a", zero_feature(feature), preserve_error=True)(rows, memo)
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(memo["a"], codes)
        # Through the reconstruction instead, the error term is lost as well.
        recon_path = sae_edit(sae, "b", zero_feature(feature))(rows, {})
        torch.testing.assert_close(recon_path, sae.decode_codes(ablated, params))
        # Joint ablation zeroes every listed feature, duplicates ignored.
        both = codes.clone()
        both[:, [feature, 0]] = 0
        joint = sae_edit(sae, "c", zero_features([feature, 0, feature]), preserve_error=True)(rows, {})
        torch.testing.assert_close(joint, rows + sae.decode_codes(both, params) - sae.decode_codes(codes, params))


class TestInjectionCohort:
    def gene(self, start):
        tss0 = start + 15 * 128 + 40
        return {"chrom": "chr1", "window_start": start, "window_end": start + L, "gene_name": f"G{start}",
                "strand": "+", "tss0": tss0, "tss_bin_index": 15,
                "exon_bin_weights": [{"bin_index": 15, "exon_bp": 88}, {"bin_index": 16, "exon_bp": 128}]}

    def run(self, monkeypatch, model, sae, fasta, tmp_path, promoter=1, **kwargs):
        path, _ = fasta
        tracks = {h: [{"index": 0}, {"index": 1}] for h in I.TRANSCRIPTIONAL_HEADS}
        monkeypatch.setattr(I, "eligible_genes", lambda *a, **k: [self.gene(0), self.gene(L)])
        monkeypatch.setattr(I, "select_tracks", lambda strand: tracks)
        return I.run_gene_injection_cohort(
            sae=sae, model=model, tap=_Tap, manifest=None, annotations=None, gtf_path=None,
            fasta_path=path, promoter_feature_id=promoter, control_feature_id=2,
            organism=torch.tensor([0]), out_dir=tmp_path, **kwargs)

    def tss_codes(self, model, sae, seq, start):
        x = torch.from_numpy(I.sequence_to_onehot(seq[start:start + L]).astype(np.float32))[None]
        return sae.encode_codes(capture_residual(model, 2, x)[0])[0][15]

    def test_dose_response_is_batching_invariant(self, monkeypatch, model, sae, fasta, tmp_path):
        runs = {}
        for max_batch in (3, 10):
            genes, stats = self.run(monkeypatch, model, sae, fasta, tmp_path / str(max_batch),
                                    max_batch=max_batch, dose=2.0, dose_multipliers=(0.5, 1, 4),
                                    preserve_error=True, dose_source="test", injection="add")
            runs[max_batch] = (genes, stats)
        assert json.dumps(runs[3], sort_keys=True) == json.dumps(runs[10], sort_keys=True)
        genes, stats = runs[3]
        assert list(stats["dose_response"]) == ["x0.5", "x1", "x4"]
        assert [v["level"] for v in stats["dose_response"].values()] == [1.0, 2.0, 8.0]
        assert stats["baseline"] == "original" and stats["dose"]["source"] == "test"
        assert stats["modalities"] == stats["dose_response"]["x1"]["modalities"]
        assert stats["n_measured_every_dose"] == 2
        for g in genes:
            assert set(g["doses"]) == {"x0.5", "x1", "x4"}
            # Adding is the same code change for every gene.
            assert [d["code_change"] for d in g["doses"].values()] == [1.0, 2.0, 8.0]
            assert g["effects"] == g["doses"]["x1"]["effects"]
            assert g["doses"]["x4"]["edit_norms"]["target"]["realized"] > 0

    def test_clamping_below_the_current_value_is_unmeasurable(self, monkeypatch, model, sae, fasta,
                                                              tmp_path):
        _path, seq = fasta
        first, second = self.tss_codes(model, sae, seq, 0), self.tss_codes(model, sae, seq, L)
        promoter = int(torch.argmax(torch.minimum(first, second)))
        a0 = float(first[promoter])
        assert a0 > 0 and float(second[promoter]) > 0
        genes, stats = self.run(monkeypatch, model, sae, fasta, tmp_path, promoter=promoter,
                                dose=a0, dose_multipliers=(0.5, 1, 4), injection="clamp")
        g = genes[0]
        assert g["feature_at_tss"] == pytest.approx(a0)
        assert g["doses"]["x0.5"]["status"] == "unmeasurable" and g["doses"]["x0.5"]["code_change"] == 0
        assert g["status"] == g["doses"]["x1"]["status"] == "unmeasurable"
        assert "reason" in g and "effects" not in g
        assert g["doses"]["x4"]["status"] == "measured"
        assert g["doses"]["x4"]["code_change"] == pytest.approx(3 * a0)
        assert stats["n_measured_every_dose"] < 2
        assert (stats["dose_response_common"]["x4"]["n_measured"] == stats["n_measured_every_dose"]
                <= stats["dose_response"]["x4"]["n_measured"])

    def test_calibration_windows_are_excluded(self, monkeypatch, model, sae, fasta, tmp_path):
        genes, stats = self.run(monkeypatch, model, sae, fasta, tmp_path,
                                exclude_windows=[{"chrom": "chr1", "start": 0, "end": 100 + 15 * 128}])
        assert [g["gene_name"] for g in genes] == [f"G{L}"]
        assert stats["n_excluded_calibration"] == 1

    def test_legacy_defaults_keep_one_fixed_dose_through_the_reconstruction(self, monkeypatch, model,
                                                                            sae, fasta, tmp_path):
        genes, stats = self.run(monkeypatch, model, sae, fasta, tmp_path)
        assert stats["dose"]["level"] == 5.0 and list(stats["dose_response"]) == ["x1"]
        assert stats["baseline"] == "reconstruction" and stats["dose"]["injection"] == "clamp"

    @pytest.mark.parametrize("multipliers", [(2.0,), (1.0, 1.0000001), (1.0, -1.0)])
    def test_bad_multipliers(self, monkeypatch, model, sae, fasta, tmp_path, multipliers):
        with pytest.raises(ValueError):
            self.run(monkeypatch, model, sae, fasta, tmp_path, dose_multipliers=multipliers)


class TestProtocol:
    def test_legacy_is_the_first_campaign(self):
        legacy = I.Protocol.legacy()
        assert (legacy.score, legacy.top_n, legacy.fixed_dose, legacy.dose_multipliers,
                legacy.injection, legacy.preserve_error) == ("mean_diff", 1, 5.0, (1.0,), "clamp", False)
        assert I.Protocol().fixed_dose is None and 1.0 in I.Protocol().dose_multipliers
        assert I.Protocol().injection == "add" and I.Protocol().triad_primary == "heads_error_preserving"
        assert json.dumps(I.Protocol().to_json())

    def parse(self, *argv):
        import argparse
        parser = argparse.ArgumentParser()
        I.add_protocol_arguments(parser)
        return I.protocol_from_args(parser.parse_args(list(argv)))

    def test_cli_overrides(self):
        assert self.parse() == I.Protocol()
        assert self.parse("--protocol", "legacy") == I.Protocol.legacy()
        p = self.parse("--protocol", "legacy", "--natural-dose", "--dose-multipliers", "1", "2")
        assert p.name == "legacy" and p.fixed_dose is None and p.dose_multipliers == (1.0, 2.0)
        # cohen_d under legacy keeps its fire-rate guard unless one is given.
        assert self.parse("--protocol", "legacy", "--score", "cohen_d").min_fire_rate == 0.05
        assert self.parse("--protocol", "legacy", "--score", "cohen_d",
                          "--min-fire-rate", "0").min_fire_rate == 0.0
        assert self.parse("--no-preserve-error", "--injection", "clamp").preserve_error is False
        with pytest.raises(SystemExit):
            self.parse("--fixed-dose", "3", "--natural-dose")

    def test_folders_are_protocol_specific(self):
        assert I.campaign_name("sae", "legacy") == "campaign_matrix"
        assert I.campaign_name("sae_v2", "legacy") == "campaign_matrix_sae_v2"
        assert I.campaign_name("sae_v2", "v2") == "campaign_matrix_sae_v2_protocol_v2"
        assert M.matrix_subdir("sae_v2", "v2") == "motif_matrix_v2_protocol_v2"

    def test_resume_refuses_another_protocol(self, tmp_path):
        I.check_resumable(tmp_path, I.Protocol())                  # empty folder
        (tmp_path / "regulatory_triad_summary.json").write_text('{"a": 1}')
        I.check_resumable(tmp_path, I.Protocol.legacy())           # first-campaign files
        with pytest.raises(RuntimeError):
            I.check_resumable(tmp_path, I.Protocol())
        (tmp_path / "feature_discovery.json").write_text(json.dumps({"protocol": I.Protocol().to_json()}))
        I.check_resumable(tmp_path, I.Protocol())
        with pytest.raises(RuntimeError):
            I.check_resumable(tmp_path, I.Protocol(top_n=5))

    def test_matrix_forwards_the_protocol(self, capsys, tmp_path):
        from ag_sae import matrix
        matrix.cmd_matrix(["--fold", "fold3", "--sae-pool", "sae_v2", "--layers", "resid_pre_b8",
                     "--seeds", "0", "--dry-run", "--protocol", "legacy", "--out-dir", str(tmp_path)])
        assert "--protocol legacy" in capsys.readouterr().out


class TestStreamCheck:
    def test_identical_streams_agree_perfectly(self, sae):
        from ag_sae.stream_check import compare_streams
        rows = torch.randn(40, DIM) * 2
        out = compare_streams(sae, rows, rows.clone())
        assert out["rel_l2"] == 0 and out["support_jaccard"] == 1 and out["code_rel_l1"] == 0
        assert out["fvu_encode"] == out["fvu_predict"]

    def test_finds_no_gap_without_autocast_and_one_with_it(self, sae, fasta):
        from ag_sae.stream_check import pooled, run_stream_check
        _path, seq = fasta
        sequences = [seq[i * L:(i + 1) * L] for i in range(2)]
        exact = FakeAlphaGenome(dim=DIM, n_blocks=4).eval().requires_grad_(False)
        report = run_stream_check(exact, sae, _Tap, sequences)
        # encode_stream rounds to float16 as the shards did; that is the only gap.
        assert report["pooled"]["rel_l2"] < 1e-3
        assert report["pooled"]["windows"] == 2
        assert report["pooled"]["onehot_max_abs_diff"] == 0
        assert report["pooled"]["mha_vs_block_entry"] == 0
        assert report["pooled"]["predict_dtype"] == ["float32"]
        bf16 = FakeAlphaGenome(dim=DIM, n_blocks=4, autocast=True).eval().requires_grad_(False)
        assert run_stream_check(bf16, sae, _Tap, sequences)["pooled"]["rel_l2"] > report["pooled"]["rel_l2"]
        assert pooled([{"rows": 1, "a_min": 1.0, "b": 0.0}, {"rows": 3, "a_min": 0.5, "b": 1.0}]) == \
            {"rows": 4, "windows": 2, "a_min": 0.5, "b": 0.75}


class TestShardCheck:
    def test_rows_stored_from_the_predict_path_match_it(self, sae, fasta):
        from ag_sae.stream_check import predict_stream, run_shard_check
        _path, seq = fasta
        model = FakeAlphaGenome(dim=DIM, n_blocks=4).eval().requires_grad_(False)
        frames, stored = [], []
        for w in range(2):
            start = w * L
            x = torch.from_numpy(I.sequence_to_onehot(seq[start:start + L]).astype(np.float32))[None]
            stream = predict_stream(model, _Tap, x)[0]
            keep = np.sort(np.random.default_rng(w).choice(BINS, 10, replace=False))
            stored.append(stream[keep].half().numpy())
            frames.append(pd.DataFrame({"chrom": "chr1", "bin_start": start + keep * 128,
                                        "window_start": start}))
        coords = pd.concat(frames, ignore_index=True)
        out = run_shard_check(model, sae, _Tap, coords, np.concatenate(stored),
                              lambda c, s, e: seq[s:e], n_windows=2, window_bp=L)
        assert out["predict"]["pooled"]["rel_l2"] == 0 and out["predict"]["pooled"]["rows"] == 20
        assert out["encode"]["pooled"]["rel_l2"] < 1e-3


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


class TestWindowsAndNulls:
    def manifest(self):
        rows = [{"chrom": c, "win_start": i * 2_000_000, "win_end": i * 2_000_000 + 1_048_576, "split": "test"}
                for c in ("chr1", "chr2", "chr3", "chr4") for i in range(30)]
        rows += [{"chrom": "chr9", "win_start": 0, "win_end": 1_048_576, "split": "train"}]
        return pd.DataFrame(rows)

    def test_random_windows_are_reproducible_spread_and_sorted(self):
        m = self.manifest()
        a = I.evaluation_windows(m, 40, "random", 7)
        assert a == I.evaluation_windows(m, 40, "random", 7) != I.evaluation_windows(m, 40, "random", 8)
        assert len({w["chrom"] for w in a}) == 4 and len(a) == 40
        assert a == sorted(a, key=lambda w: (w["chrom"], w["start"]))
        assert I.evaluation_windows(m, 3, "first") == [
            {"chrom": "chr1", "start": s, "end": s + 1_048_576} for s in (0, 2_000_000, 4_000_000)]

    def test_matched_pools_exclude_concept_features_and_match_rates(self):
        rate = np.linspace(0.0, 0.5, 64)
        feats = I.DiscoveredFeatures(10, 20, 30, 40, ranked={"cCRE_PLS": (10, 11, 12)}, firing_rate=rate)
        pools = I.matched_null_pools(feats, top_n=3, pool_size=5)
        assert set(pools) == {"PLS_Promoter", "CTCF_Insulator", "Enhancer_pELS", "Shuffled_Control"}
        banned = {10, 11, 12, 20, 30, 40, 0}          # top-n of every concept, and a feature that never fires
        assert not banned & set().union(*map(set, pools.values()))
        assert set(pools["PLS_Promoter"]) == {7, 8, 9, 13, 14}
        draw = I.draw_null_features(pools["PLS_Promoter"], 2, seed=1, window_index=3, concept_index=0)
        assert draw == I.draw_null_features(pools["PLS_Promoter"], 2, 1, 3, 0) and len(set(draw)) == 2
        with pytest.raises(ValueError):
            I.matched_null_pools(I.DiscoveredFeatures(1, 2, 3, 4), 1)

    def test_triad_with_nulls_and_baselines(self, model, sae, fasta, tmp_path):
        path, seq = fasta
        windows = [{"chrom": "chr1", "start": 0, "end": L}, {"chrom": "chr1", "start": L, "end": 2 * L}]
        ann = pd.concat([TestTriad().annotations(w["start"]) for w in windows], ignore_index=True)
        found = I.discover_concept_features(sae, model, _Tap, windows, ann, path, torch.tensor([0]),
                                            top_n=2, min_fire_rate=0.0)
        assert found.firing_rate is not None and found.firing_rate.shape == (64,)
        # The fake SAE fires only a handful of features, all of them concept
        # features; give every feature a rate so the pools are not empty.
        import dataclasses
        found = dataclasses.replace(found, firing_rate=np.linspace(0.01, 0.5, 64))
        results, summary = I.run_triad_ablation(sae, model, _Tap, windows, ann, found, path,
                                                torch.tensor([0]), tmp_path, top_n=2,
                                                null_per_window=2, null_pool_size=6, null_seed=3)
        entry = results[0]["features"]["PLS_Promoter"]["heads_error_preserving"]["atac"]
        assert entry["original_on_concept"] == pytest.approx(entry["baseline_on_concept"])
        nulls = results[0]["null"]["PLS_Promoter"]
        assert len(nulls) == 2 and all(n["feature_id"] in summary["PLS_Promoter"]["null"]["pool"] for n in nulls)
        diff = summary["PLS_Promoter"]["null"]["concept_minus_null"]["atac"]
        expected = np.mean([w["features"]["PLS_Promoter"]["heads_error_preserving"]["atac"]["delta_on_concept"]
                            - np.mean([n["heads_error_preserving"]["atac"]["delta_on_concept"]
                                       for n in w["null"]["PLS_Promoter"]]) for w in results])
        assert diff["mean_delta"] == pytest.approx(expected)

    def test_protocols_pick_windows_and_nulls(self):
        assert I.Protocol().window_sampling == "random" and I.Protocol().null_per_window == 1
        legacy = I.Protocol.legacy()
        assert legacy.window_sampling == "first" and legacy.null_per_window == 0


class TestWindowSplit:
    def test_calibration_is_disjoint_reproducible_and_legacy_compatible(self):
        m = TestWindowsAndNulls().manifest()
        ev, cal = I.split_windows(m, I.Protocol(n_windows=40))
        assert (ev, cal) == I.split_windows(m, I.Protocol(n_windows=40)) and len(cal) == 10
        assert not {(w["chrom"], w["start"]) for w in ev} & {(w["chrom"], w["start"]) for w in cal}
        assert len({w["chrom"] for w in cal}) > 1                     # random, not the first ten
        lev, lcal = I.split_windows(m, I.Protocol.legacy())
        test = m[m.split == "test"]
        assert [w["start"] for w in lev] == test.win_start.head(40).tolist()
        assert [w["start"] for w in lcal] == test.win_start.iloc[40:50].tolist()

    def test_motif_candidates_skip_calibration_windows(self):
        m = TestWindowsAndNulls().manifest()
        _ev, cal = I.split_windows(m, I.Protocol())
        cands = M.motif_candidates(m, "v2")
        assert cands == M.motif_candidates(m, "v2") and len(cands) == 120 - 10
        assert not {(w["chrom"], w["start"]) for w in cal} & {(w["chrom"], w["start"]) for w in cands}
        assert M.motif_candidates(m, "legacy")[:2] == [{"chrom": "chr1", "start": 0, "end": 1_048_576},
                                                        {"chrom": "chr1", "start": 2_000_000,
                                                         "end": 2_000_000 + 1_048_576}]


class TestDiscoveryFiringRates:
    def test_rates_count_only_edited_bins_and_runners_up_are_recorded(self, monkeypatch, tmp_path):
        codes = torch.zeros(BINS, 64)
        codes[[3, 4], 5] = 2.0            # fires on the PLS bins (edited)
        codes[:, 9] = 0.5                 # fires everywhere
        codes[[30, 31], 12] = 1.0         # fires only on bins that are never edited
        monkeypatch.setattr(I, "_window_codes", lambda *a, **k: codes)
        ann = TestTriad().annotations(0)
        found = I.discover_concept_features(None, None, _Tap, [{"chrom": "chr1", "start": 0, "end": L}],
                                            ann, tmp_path, torch.tensor([0]), min_fire_rate=0.0)
        edited = I.edited_bins_mask(ann, 0, BINS, 128)
        assert found.firing_rate[5] == pytest.approx(2 / edited.sum())
        assert found.firing_rate[9] == pytest.approx(1.0) and found.firing_rate[12] == 0.0
        assert found.promoter_feature_id == 5 and 5 in found.near_top
