"""The results tables and the hierarchical analysis, on synthetic campaigns with known answers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from ag_sae import analyze as A
from ag_sae import results as R

HEADS = ("atac", "chip_tf")
CONCEPTS = ("PLS_Promoter", "CTCF_Insulator")


def head_block(delta, baseline):
    return {h: {"delta_on_concept": delta[h], "baseline_on_concept": baseline,
                "original_on_concept": baseline, "has_concept_bins": True,
                "delta_on_all_regulatory": 0.0} for h in HEADS}


def write_campaign(root, folds=("fold0", "fold1", "fold2", "fold3"), seeds=("seed0", "seed1", "seed2"),
                   n_windows=12, effect=None, null_effect=-0.01, rng=None):
    """Triad + injection + motif folders where PLS ablation lowers atac by `effect` (relative)."""
    rng = rng or np.random.default_rng(0)
    effect = effect or {"fold0": -0.2, "fold1": -0.25, "fold2": -0.15, "fold3": -0.3}
    campaign, motif = root / "campaign", root / "motif"
    for fold in folds:
        for seed in seeds:
            name = f"{fold}_resid_pre_b8_{seed}"
            windows = []
            for w in range(n_windows):
                chrom, start = f"chr{1 + w % 3}", 20_000_000 * w
                base = 10.0
                pls = {"atac": base * (effect[fold] + rng.normal(0, 0.02)), "chip_tf": base * rng.normal(0, 0.01)}
                ctcf = {"atac": base * rng.normal(0, 0.01), "chip_tf": base * (-0.4 + rng.normal(0, 0.02))}
                nulls = {c: [{"feature_id": 99, "fired_bins": 3, "heads_error_preserving": head_block(
                    {h: base * (null_effect + rng.normal(0, 0.005)) for h in HEADS}, base)}] for c in CONCEPTS}
                windows.append({
                    "window_index": w, "chrom": chrom, "start": start,
                    "features": {
                        "PLS_Promoter": {"feature_id": 1, "fired_bins": 5,
                                         "heads": head_block(pls, base), "heads_error_preserving": head_block(pls, base)},
                        "CTCF_Insulator": {"feature_id": 2, "fired_bins": 5,
                                           "heads": head_block(ctcf, base), "heads_error_preserving": head_block(ctcf, base)}},
                    "joint": {}, "null": nulls})
            folder = campaign / name
            folder.mkdir(parents=True)
            (folder / R.TRIAD_FILE).write_text(json.dumps(windows))
            genes = []
            for g in range(10):
                doses = {}
                for m in (0.5, 1.0, 2.0):
                    eff = {mod: {"n_tracks": 3, "target_median_log2_ratio": 0.0, "control_median_log2_ratio": 0.0,
                                 "paired_target_minus_control_median": (0.1 * m if mod == "cage" else 0.0)
                                 + rng.normal(0, 0.01),
                                 "paired_difference": 0.0, "fraction_target_increase": 0.5}
                           for mod in ("cage", "rna_seq")}
                    doses[R_tag(m)] = {"level": m, "code_change": m, "status": "measured", "effects": eff}
                genes.append({"rank": g, "gene_name": f"G{g}", "chrom": f"chr{1 + g % 3}", "tss0": 15_000_000 * g,
                              "feature_at_tss": 0.0, "status": "measured", "doses": doses,
                              "effects": doses["x1"]["effects"]})
            (folder / R.GENE_FILE).write_text(json.dumps(genes))
            (folder / "feature_discovery.json").write_text(json.dumps(
                {"primary": {"PLS_Promoter": 1, "CTCF_Insulator": 2}, "injection_dose": {"level": 5.0},
                 "protocol": {"name": "v2"}}))
            mfolder = motif / name
            mfolder.mkdir(parents=True)
            mw = [{"chrom": f"chr{1 + w % 3}", "start": 20_000_000 * w, "wt_latent_val": 1.0, "mut_latent_val": 0.0,
                   "heads": {"chip_tf": {"y_wt": 100.0, "y_ism": 50.0, "y_abl": 95.0, "y_rescue": 50.0,
                                         "delta_ism": -50.0, "delta_abl": -5.0, "delta_rescue": 0.0}}}
                  for w in range(n_windows)]
            (mfolder / R.MOTIF_FILE).write_text(json.dumps(mw))
    return campaign, motif


def R_tag(m):
    return f"x{m:g}"


@pytest.fixture()
def campaign(tmp_path):
    return write_campaign(tmp_path)


class TestResults:
    def test_tables_have_one_row_per_measurement(self, campaign):
        c, m = campaign
        tables = R.collect({"sae_v2": c}, {"sae_v2": m})
        triad, inj, motif = tables["triad"], tables["injection"], tables["motif"]
        # 4 folds x 3 seeds x 12 windows x 2 concepts x (2 methods + 1 null) x 2 heads
        assert len(triad) == 4 * 3 * 12 * 2 * 3 * 2
        row = triad[(triad.kind == "feature") & (triad.method == "error_preserving")].iloc[0]
        assert row.relative_change == pytest.approx(row.delta / row.baseline)
        assert set(inj.dose) == {"x0.5", "x1", "x2"} and set(inj.modality) == {"cage", "rna_seq"}
        assert len(motif) == 4 * 3 * 12 and set(tables["discovery"].concept) == set(CONCEPTS)

    def test_first_campaign_gene_records_read_as_one_dose(self):
        rows = R.gene_rows([{"gene_name": "A", "status": "measured", "effects": {"cage": {
            "n_tracks": 1, "target_median_log2_ratio": 0.1, "control_median_log2_ratio": 0.0,
            "paired_target_minus_control_median": 0.1, "paired_difference": 1.0,
            "fraction_target_increase": 1.0}}}], {"pool": "sae"})
        assert len(rows) == 1 and rows[0]["dose"] == "x1"

    def test_unrecognised_folders_are_ignored(self, tmp_path):
        (tmp_path / "notes").mkdir()
        assert R.collect({"p": tmp_path})["triad"].empty


class TestStatistics:
    def test_bh_matches_scipy_and_keeps_nan(self):
        p = np.array([0.01, np.nan, 0.04, 0.03])
        q = A.bh_adjust(p)
        assert np.isnan(q[1])
        np.testing.assert_allclose(q[[0, 2, 3]], sps.false_discovery_control([0.01, 0.04, 0.03]))

    def test_tost_accepts_a_tight_zero_and_rejects_a_shift(self):
        rng = np.random.default_rng(0)
        assert A.tost_p(rng.normal(0, 0.01, 50), 0.1) < 0.001
        assert A.tost_p(rng.normal(0.3, 0.01, 50), 0.1) > 0.5

    def test_block_bootstrap_ratio_and_mean(self):
        x = np.arange(10, dtype=float)
        blocks = np.repeat(np.arange(5), 2)
        reps = A.block_bootstrap(x, blocks, n_boot=200, seed=1)
        assert reps.shape == (200,) and reps.min() >= 0 and reps.max() <= 9
        ratio = A.block_bootstrap(x, blocks, denominator=2 * x + 1e-9, n_boot=50, seed=1)
        np.testing.assert_allclose(ratio, 0.5, rtol=1e-6)

    def test_hierarchical_recovers_fold_means_and_agreement(self):
        rows = []
        for f, mu in enumerate((-0.2, -0.25, -0.15, 0.05)):
            for w in range(20):
                rows.append({"g": "a", "fold": f"fold{f}", "chrom": "chr1", "start": w * 20_000_000,
                             "v": mu + 0.001 * (w - 10)})
        per_fold, across = A.hierarchical(pd.DataFrame(rows), "v", ["g"], "start", n_boot=300)
        np.testing.assert_allclose(per_fold.estimate, [-0.2, -0.25, -0.15, 0.05], atol=1e-3)
        a = across.iloc[0]
        assert a.estimate == pytest.approx(np.mean([-0.2, -0.25, -0.15, 0.05]), abs=1e-3)
        assert a.n_folds == 4 and a.folds_same_sign == 3
        assert a.ci_low < a.estimate < a.ci_high
        assert "q_wilcoxon" in per_fold and "q_folds_ttest" in across


class TestAnalysis:
    def test_end_to_end(self, campaign, tmp_path):
        c, m = campaign
        tables = R.collect({"sae_v2": c}, {"sae_v2": m})
        summary = A.run_analysis(tables, tmp_path / "out", n_boot=200, figures=True)
        stats_dir = tmp_path / "out" / "stats"
        triad = pd.read_csv(stats_dir / "triad_across_folds.csv")
        pls = triad[(triad.measure == "feature_ep") & (triad.concept == "PLS_Promoter") & (triad["head"] == "atac")]
        assert pls.estimate.iloc[0] == pytest.approx(-0.225, abs=0.01)
        assert pls.folds_same_sign.iloc[0] == 4 and pls.ci_high.iloc[0] < 0 and pls.replicated.iloc[0]
        assert pls.t_ci_low.iloc[0] < pls.ci_low.iloc[0]          # the fold interval is the wider one
        spec = triad[(triad.measure == "specificity_ep") & (triad.concept == "PLS_Promoter") & (triad["head"] == "atac")]
        assert spec.estimate.iloc[0] == pytest.approx(-0.215, abs=0.01)     # minus the -0.01 null
        slope = pd.read_csv(stats_dir / "dose_slope_across_folds.csv").set_index("modality")
        # paired log2 = 0.1 * m, so per doubling the slope rises with m: about 0.0875 over 0.5..2.
        assert slope.loc["cage", "estimate"] > 0.05 and abs(slope.loc["rna_seq", "estimate"]) < 0.01
        eq = pd.read_csv(stats_dir / "injection_equivalence_per_fold.csv").set_index(["modality", "fold"])
        assert eq.loc["rna_seq"].equivalent.all() and not eq.loc["cage"].equivalent.any()
        eq_all = pd.read_csv(stats_dir / "injection_equivalence_across_folds.csv").set_index("modality")
        assert eq_all.loc["rna_seq", "p_tost_folds"] < 0.05 and eq_all.loc["rna_seq", "all_folds_equivalent"]
        med = pd.read_csv(stats_dir / "motif_mediation_across_folds.csv")
        assert med.estimate.iloc[0] == pytest.approx(0.1)
        assert any(p.endswith("triad_feature_ep_sae_v2.pdf") for p in summary["figures"])

    def test_cli_writes_and_reuses_tables(self, campaign, tmp_path):
        c, m = campaign
        out = tmp_path / "cli"
        A.main(["--campaign", f"sae_v2={c}", "--motif", f"sae_v2={m}", "--out", str(out),
                "--n-boot", "50", "--no-figures"])
        assert (out / "tables" / "triad.parquet").is_file()
        A.main(["--tables", str(out / "tables"), "--out", str(tmp_path / "again"), "--n-boot", "50",
                "--no-figures"])
        assert (tmp_path / "again" / "stats" / "triad_across_folds.csv").is_file()


class TestRobustness:
    def test_a_near_zero_baseline_cannot_dominate(self):
        rows = []
        for w in range(40):
            delta, base = (-0.002, 1e-4) if w == 0 else (-0.5, 10.0)
            rows.append({"g": "a", "fold": "fold0", "chrom": "chr1", "start": w * 20_000_000,
                         "delta": delta, "baseline": base})
        _pf, across = A.hierarchical(pd.DataFrame(rows), "delta", ["g"], "start", n_boot=100,
                                     ratio_of=("delta", "baseline"))
        assert across.estimate.iloc[0] == pytest.approx(-0.05, abs=1e-3)

    def test_a_dose_nobody_could_measure_empties_the_common_set(self, tmp_path):
        c, _m = write_campaign(tmp_path / "c", folds=("fold0",), seeds=("seed0",))
        inj = R.collect({"p": c})["injection"]
        inj.loc[inj.dose == "x0.5", "status"] = "unmeasurable"
        out = A.analyze_injection(inj, n_boot=50, seed=0, tost_margin=0.1)
        assert out["injection_common_across_folds"].empty or \
            "x0.5" not in set(out["injection_common_across_folds"].dose)
        assert len(out["injection_common_per_fold"]) == 0

    def test_runs_without_baselines_fall_back_to_delta(self, tmp_path):
        c, _m = write_campaign(tmp_path / "c", folds=("fold0", "fold1"), seeds=("seed0",))
        tables = R.collect({"p": c})
        tables["triad"]["baseline"] = np.nan
        tables["triad"]["relative_change"] = np.nan
        A.run_analysis(tables, tmp_path / "out", n_boot=50, figures=True)
        across = pd.read_csv(tmp_path / "out" / "stats" / "triad_across_folds.csv")
        assert set(across.value) == {"delta"} and len(across)

    def test_null_diagnostics_compare_edit_sizes(self, campaign):
        c, _m = campaign
        diag = A.null_diagnostics(R.collect({"p": c})["triad"])
        assert set(diag.kind) == {"feature", "null"} and (diag.zero_fire_share == 0).all()
