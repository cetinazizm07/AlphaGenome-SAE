"""Correctness tests for concept matching, especially the tie-heavy AUROC."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ag_sae import concepts as C


def brute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Definition of AUROC, written out pair by pair. O(n^2), for tests only."""
    pos, neg = scores[labels], scores[~labels]
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (pos.size * neg.size))


def grid(n_bins: int, chrom: str = "chr19", bin_bp: int = 128) -> pd.DataFrame:
    return C.bin_grid(chrom, 0, n_bins * bin_bp, bin_bp)


# --- interval labelling ---------------------------------------------------


def test_overlap_is_half_open_on_both_sides():
    bins = grid(4)  # [0,128) [128,256) [256,384) [384,512)
    spans = {"chr19": np.array([[128, 256]], dtype=np.int64)}
    assert C.label_bins(bins, spans).tolist() == [False, True, False, False]

    # Touching exactly at a boundary is not an overlap.
    assert C.label_bins(bins, {"chr19": np.array([[0, 128]])}).tolist() == [True, False, False, False]
    # One base of overlap is an overlap.
    assert C.label_bins(bins, {"chr19": np.array([[127, 129]])}).tolist() == [True, True, False, False]


def test_overlap_handles_nested_and_unsorted_intervals():
    bins = grid(6)
    # [100,500] covers bins 0-3, [200,260] is nested inside it, [660,700] hits
    # only bin 5 -- so bin 4 must come out negative even though intervals sit
    # on both sides of it.
    spans = {"chr19": np.array([[660, 700], [100, 500], [200, 260]], dtype=np.int64)}
    spans["chr19"] = spans["chr19"][np.argsort(spans["chr19"][:, 0])]
    expected = [True, True, True, True, False, True]
    assert C.label_bins(bins, spans).tolist() == expected


def test_other_chromosomes_never_leak():
    bins = pd.concat([C.bin_grid("chr1", 0, 256, 128), C.bin_grid("chr2", 0, 256, 128)],
                     ignore_index=True)
    got = C.label_bins(bins, {"chr1": np.array([[0, 256]])})
    assert got.tolist() == [True, True, False, False]


# --- AUROC ----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_sparse_auroc_matches_the_definition(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(40, 120)
    dense = np.where(rng.random(n) < 0.3, rng.random(n) + 0.01, 0.0)
    labels = rng.random(n) < 0.35
    if labels.all() or not labels.any():
        pytest.skip("degenerate draw")
    nz = dense > 0
    got = C.auroc_from_nonzero(dense[nz], labels[nz], int(labels.sum()), int((~labels).sum()))
    assert got == pytest.approx(brute_auroc(dense, labels))


def test_sparse_auroc_edge_cases():
    labels = np.array([True, True, False, False])
    # Fires only on positives -> perfect.
    assert C.auroc_from_nonzero(np.array([1.0, 2.0]), np.array([True, True]), 2, 2) == 1.0
    # Fires only on negatives -> zero.
    assert C.auroc_from_nonzero(np.array([1.0, 2.0]), np.array([False, False]), 2, 2) == 0.0
    # Never fires -> chance, not an error and not NaN.
    assert C.auroc_from_nonzero(np.empty(0), np.empty(0, dtype=bool), 2, 2) == 0.5
    # All bins tied at one value -> chance.
    assert C.auroc_from_nonzero(np.ones(4), labels, 2, 2) == 0.5


def test_sparse_auroc_rejects_bad_input():
    with pytest.raises(ValueError, match="strictly positive"):
        C.auroc_from_nonzero(np.array([0.0]), np.array([True]), 1, 1)
    with pytest.raises(ValueError, match="at least one positive"):
        C.auroc_from_nonzero(np.empty(0), np.empty(0, dtype=bool), 0, 4)
    with pytest.raises(ValueError, match="More nonzero"):
        C.auroc_from_nonzero(np.ones(3), np.array([True, True, True]), 2, 5)


def test_dead_features_score_chance_not_missing():
    codes = sparse.csc_matrix(np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 0.5]]))
    labels = np.array([True, False, True])
    got = C.auroc_all_features(codes, labels)
    assert got[0] == 0.5
    assert got[1] == pytest.approx(brute_auroc(np.array([1.0, 2.0, 0.5]), labels))


@pytest.mark.parametrize("seed", range(6))
def test_dense_auroc_matches_definition_with_signed_scores(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 5))
    x[:, 0] = np.round(x[:, 0])  # force ties into one column
    labels = rng.random(60) < 0.4
    got = C.auroc_dense(x, labels)
    for column in range(x.shape[1]):
        assert got[column] == pytest.approx(brute_auroc(x[:, column], labels))


def test_dense_auroc_survives_float16_shards():
    # Shards are stored float16. scipy's rankdata keeps the input dtype, so
    # ranks above 2048 lose precision and their sum overflows to inf. The
    # result must match the float64 answer exactly.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 3))
    labels = rng.random(5000) < 0.3
    reference = C.auroc_dense(x.astype(np.float64), labels)
    got = C.auroc_dense(x.astype(np.float16), labels)
    assert np.isfinite(got).all()
    assert got == pytest.approx(reference, abs=2e-3)


def test_sparse_auroc_survives_float16_values():
    rng = np.random.default_rng(1)
    values = rng.random(5000).astype(np.float16)
    is_positive = rng.random(5000) < 0.5
    got = C.auroc_from_nonzero(values, is_positive, n_pos=3000, n_neg=4000)
    assert np.isfinite(got)


def test_wrong_bin_width_is_rejected(monkeypatch):
    # A conv tap has 4 bp bins. Matching it with the 128 bp default would make
    # every bin its own run and silently neuter the null.
    rng = np.random.default_rng(0)
    bins = C.bin_grid("chr1", 0, 4 * 600, 4)
    acts = rng.normal(size=(len(bins), 6)).astype(np.float32)
    sae = C.FrozenSAE(
        W_enc=rng.normal(size=(6, 12)).astype(np.float32),
        b_enc=np.zeros(12, dtype=np.float32),
        b_pre=np.zeros(6, dtype=np.float32),
        k=3,
    )
    ccre = {"PLS": {"chr1": np.array([[0, 400]], dtype=np.int64)}}
    with pytest.raises(ValueError, match="4 bp wide"):
        C.match_concepts(sae, acts, bins, ccre, bin_bp=128, n_permutations=2)


def test_directed_folds_and_reports_direction():
    folded, inverse = C.directed(np.array([0.9, 0.1, 0.5]))
    assert folded == pytest.approx([0.9, 0.9, 0.5])
    assert inverse.tolist() == [False, True, False]


# --- null -----------------------------------------------------------------


def test_shift_moves_intervals_and_keeps_their_shape():
    spans = np.array([[10, 20], [40, 50]], dtype=np.int64)
    moved = C.shift_intervals(spans, 0, 100, offset=25)
    assert moved.tolist() == [[35, 45], [65, 75]]
    # Count, widths and the gap between elements all survive.
    assert (moved[:, 1] - moved[:, 0]).tolist() == [10, 10]
    assert moved[1, 0] - moved[0, 0] == spans[1, 0] - spans[0, 0]


def test_shift_wraps_at_the_far_edge():
    spans = np.array([[90, 100]], dtype=np.int64)
    moved = C.shift_intervals(spans, 0, 100, offset=5)
    # Crosses the end, so it returns as two pieces with the same total length.
    assert (moved[:, 1] - moved[:, 0]).sum() == 10
    assert moved[:, 0].min() >= 0 and moved[:, 1].max() <= 100


def test_shift_is_a_no_op_at_zero_offset():
    spans = np.array([[10, 20], [40, 50]], dtype=np.int64)
    assert C.shift_intervals(spans, 0, 100, 0).tolist() == spans.tolist()


def test_domains_follow_the_extraction_window_when_present():
    bins = pd.concat([C.bin_grid("chr1", 0, 512, 128),
                      C.bin_grid("chr1", 4096, 4608, 128)], ignore_index=True)
    bins["window_start"] = [0] * 4 + [4096] * 4
    domains = C.concept_domains(bins)
    assert [(d[0], d[1]) for d in domains] == [("chr1", 0), ("chr1", 4096)]
    # Without the column the whole chromosome is one domain.
    plain = C.concept_domains(bins.drop(columns="window_start"))
    assert len(plain) == 1


def test_the_null_still_moves_labels_when_bins_are_scattered():
    # The failure this null was written to fix. Conv taps keep a sparse random
    # subset of a window's bins, so neighbouring rows are far apart and a shift
    # among rows does nothing. Shifting the concept has to work anyway.
    rng = np.random.default_rng(0)
    window, bin_bp = 1_048_576, 64
    kept = np.sort(rng.choice(window // bin_bp, 8192, replace=False)) * bin_bp
    bins = pd.DataFrame({"chrom": "chr1", "bin_start": kept,
                         "bin_end": kept + bin_bp, "window_start": 0})
    spans = np.sort(rng.choice(window - 400, 80, replace=False))
    intervals = {"chr1": np.stack([spans, spans + 300], axis=1)}
    labels = C.label_bins(bins, intervals)
    draws = C.shifted_labels(bins, intervals, 20, seed=1)
    unchanged = np.mean([float((d == labels).mean()) for d in draws])
    assert unchanged < 0.999, "the shift barely moved the labels"
    assert all(d.sum() > 0 for d in draws)


def test_shifted_draws_are_reproducible():
    bins = grid(200)
    bins["window_start"] = 0
    intervals = {"chr19": np.array([[1280, 2560]], dtype=np.int64)}
    a = C.shifted_labels(bins, intervals, 5, seed=3)
    b = C.shifted_labels(bins, intervals, 5, seed=3)
    for left, right in zip(a, b):
        assert (left == right).all()


def test_null_values_are_valid_aurocs():
    bins = grid(200)
    bins["window_start"] = 0
    intervals = {"chr19": np.array([[6400, 8960]], dtype=np.int64)}
    codes = C.RankedCodes(sparse.csc_matrix(
        np.abs(np.random.default_rng(0).normal(size=(200, 3)))))
    draws = C.shifted_labels(bins, intervals, 25, seed=1)
    null = C.circular_null(codes, draws)
    assert null.shape == (25,)
    assert ((null >= 0) & (null <= 1)).all()


# --- encoding -------------------------------------------------------------


def toy_sae(d: int = 4, hidden: int = 8, k: int = 1) -> C.FrozenSAE:
    W = np.zeros((d, hidden), dtype=np.float32)
    W[0, 0] = 1.0   # feature 0 reads channel 0
    W[1, 1] = 1.0   # feature 1 reads channel 1
    return C.FrozenSAE(W_enc=W, b_enc=np.zeros(hidden, dtype=np.float32),
                       b_pre=np.zeros(d, dtype=np.float32), k=k, token_layernorm=False)


def test_encode_respects_topk_and_nonnegativity():
    sae = toy_sae(k=2)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4)).astype(np.float32)
    codes = sae.encode(x, chunk=7)          # chunk < n to exercise stacking
    assert codes.shape == (50, 8)
    assert (codes.data > 0).all()           # ReLU + eliminate_zeros
    per_row = np.diff(codes.tocsr().indptr)
    assert (per_row <= sae.k).all()         # never more than k, may be fewer


def test_encode_rejects_wrong_width_and_bad_scale():
    sae = toy_sae()
    with pytest.raises(ValueError, match="Expected"):
        sae.encode(np.zeros((3, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="channel_scale must be positive"):
        C.FrozenSAE(W_enc=np.zeros((4, 8), dtype=np.float32), b_enc=np.zeros(8),
                    b_pre=np.zeros(4), k=1, channel_scale=np.zeros(4))


# --- end to end -----------------------------------------------------------


def test_planted_feature_is_recovered_and_beats_the_null():
    """A feature built to fire exactly on the concept must come out on top."""
    n = 400
    bins = grid(n)
    positive = np.zeros(n, dtype=bool)
    for start in range(20, n, 40):
        positive[start:start + 8] = True     # clustered, like real elements

    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, size=(n, 4)).astype(np.float32)
    x[:, 0] = np.where(positive, 5.0, 0.0)   # channel 0 carries the concept
    x[:, 1] = 0.5                            # channel 1 is the fallback winner

    spans = []
    edges = np.flatnonzero(np.diff(positive.astype(int)) == 1) + 1
    for start in edges:
        stop = start + 8
        spans.append((int(bins.bin_start[start]), int(bins.bin_end[stop - 1])))
    ccre = {"PLS": {"chr19": np.asarray(sorted(spans), dtype=np.int64)}}

    result = C.match_concepts(toy_sae(k=1), x, bins, ccre, n_permutations=40, seed=0, top_n=5)

    row = result.per_concept.set_index("concept").loc["PLS"]
    assert row.best_feature == 0
    assert row.best_auroc == pytest.approx(1.0)
    assert row.recovered
    assert row.null_p95 < 1.0
    assert result.summary["n_features_fired"] == 2      # only features 0 and 1 ever fire
    assert result.summary["mean_l0"] == pytest.approx(1.0)

    top = result.top_bins[result.top_bins.feature == 0]
    assert top.PLS.all()                                 # every top bin is annotated
    dominant = result.per_feature.set_index("feature").loc[0]
    assert dominant.dominant == "PLS" and dominant.dominant_share == 1.0


def test_rare_concepts_are_skipped_not_silently_scored():
    bins = grid(300)
    x = np.random.default_rng(0).normal(size=(300, 4)).astype(np.float32)
    ccre = {"rare": {"chr19": np.array([[0, 128]], dtype=np.int64)}}
    result = C.match_concepts(toy_sae(), x, bins, ccre, n_permutations=5)
    assert result.per_concept.empty
    assert result.summary["concepts_skipped_too_rare"] == {"rare": 1}


def test_mismatched_rows_are_rejected():
    with pytest.raises(ValueError, match="same rows"):
        C.match_concepts(toy_sae(), np.zeros((10, 4), dtype=np.float32), grid(200), {})


# --- IO -------------------------------------------------------------------


def test_ccre_bed_reader_splits_multiclass_rows(tmp_path):
    path = tmp_path / "ccre.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("# comment\n")
        handle.write("chr19\t100\t300\trDHS1\tEH1\tPLS\n")
        handle.write("chr19\t500\t700\trDHS2\tEH2\tdELS,CTCF-only\n")
        handle.write("chr20\t100\t200\trDHS3\tEH3\t.\n")
    parsed = C.read_ccre_bed(path)
    assert set(parsed) == {"PLS", "dELS", "CTCF-only"}
    assert parsed["dELS"]["chr19"].tolist() == [[500, 700]]
    assert parsed["CTCF-only"]["chr19"].tolist() == [[500, 700]]


def test_ccre_bed_rejects_invalid_interval(tmp_path):
    path = tmp_path / "bad.bed"
    path.write_text("chr19\t300\t100\tx\ty\tPLS\n")
    with pytest.raises(ValueError, match="invalid interval"):
        C.read_ccre_bed(path)


@pytest.mark.parametrize("text,expected", [
    ("chr19:1000-2000", ("chr19", 1000, 2000)),
    ("chr1:1,000,000-2,000,000", ("chr1", 1000000, 2000000)),
])
def test_parse_region(text, expected):
    assert C.parse_region(text) == expected


def test_parse_region_rejects_garbage():
    with pytest.raises(ValueError, match="Malformed region"):
        C.parse_region("chr19")


# --- fast path ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_ranked_codes_match_the_reference_implementation(seed):
    """The vectorised path must agree with the readable loop, bit for bit."""
    rng = np.random.default_rng(seed)
    n_bins, n_features = 120, 30
    dense = np.where(rng.random((n_bins, n_features)) < 0.2, rng.random((n_bins, n_features)) + 0.01, 0.0)
    dense[:, 3] = 0.0                      # a dead feature
    dense[:, 4] = 0.7                      # a feature that fires everywhere, all tied
    codes = sparse.csc_matrix(dense)
    labels = rng.random(n_bins) < 0.3

    fast = C.RankedCodes(codes).auroc(labels)
    reference = C.auroc_all_features(codes, labels)
    np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-12)
    assert fast[3] == 0.5
    for column in (0, 1, 2, 4):
        assert fast[column] == pytest.approx(brute_auroc(dense[:, column], labels))


def test_ranked_codes_handles_leading_and_trailing_dead_features():
    dense = np.zeros((20, 5))
    dense[:, 2] = np.linspace(0.1, 1.0, 20)   # only the middle feature fires
    ranked = C.RankedCodes(sparse.csc_matrix(dense))
    labels = np.zeros(20, dtype=bool)
    labels[10:] = True
    got = ranked.auroc(labels)
    assert got.tolist()[:2] == [0.5, 0.5]
    assert got.tolist()[3:] == [0.5, 0.5]
    assert got[2] == pytest.approx(1.0)
    assert ranked.fired.tolist() == [False, False, True, False, False]


class TestPairedNulls:
    def _setup(self, n_bins=600, d_in=40, n_features=400, seed=0):
        rng = np.random.default_rng(seed)
        bins = C.bin_grid("chr1", 0, 128 * n_bins, 128)
        acts = rng.normal(size=(n_bins, d_in)).astype(np.float32)
        sae = C.FrozenSAE(
            W_enc=rng.normal(size=(d_in, n_features)).astype(np.float32) * 0.3,
            b_enc=np.zeros(n_features, dtype=np.float32),
            b_pre=np.zeros(d_in, dtype=np.float32),
            k=20,
        )
        return bins, acts, sae

    def _draws(self, n, k, count=40, seed=0):
        """`count` shifted label vectors over `n` bins with `k` positives."""
        bins = grid(n)
        bins["window_start"] = 0
        width = k * 128
        intervals = {"chr19": np.array([[0, width]], dtype=np.int64)}
        return bins, C.shifted_labels(bins, intervals, count, seed=seed)

    def test_shifts_are_identical_for_the_same_seed(self):
        _, a = self._draws(200, 20, count=5, seed=3)
        _, b = self._draws(200, 20, count=5, seed=3)
        for left, right in zip(a, b):
            assert (left == right).all()
        # The element keeps its length, so every draw has the same count.
        assert len({int(s.sum()) for s in a}) == 1

    def test_more_candidates_give_a_higher_null_ceiling(self):
        # Under pure noise the best of many columns beats the best of few, so
        # an uncalibrated comparison favours whichever side has more.
        rng = np.random.default_rng(1)
        _, draws = self._draws(400, 80, count=40, seed=0)
        narrow = rng.normal(size=(400, 8)).astype(np.float32)
        wide = rng.normal(size=(400, 512)).astype(np.float32)
        ceiling_narrow = np.quantile(C.circular_null_dense(narrow, draws), 0.95)
        ceiling_wide = np.quantile(C.circular_null_dense(wide, draws), 0.95)
        assert ceiling_wide > ceiling_narrow + 0.02

    def test_dense_null_matches_a_direct_computation(self):
        rng = np.random.default_rng(2)
        _, draws = self._draws(200, 60, count=6, seed=5)
        acts = rng.normal(size=(200, 12)).astype(np.float32)
        got = C.circular_null_dense(acts, draws, block=5)
        want = []
        for shifted in draws:
            folded, _ = C.directed(C.auroc_dense(acts, shifted))
            want.append(folded.max())
        assert got == pytest.approx(want)

    def test_degenerate_draws_score_chance_instead_of_crashing(self):
        # A concept too small to survive the shift must not kill the run.
        empty = [np.zeros(50, dtype=bool), np.ones(50, dtype=bool)]
        acts = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
        assert C.circular_null_dense(acts, empty).tolist() == [0.5, 0.5]

    def test_matcher_reports_both_ceilings_and_the_calibrated_gap(self, tmp_path):
        bins, acts, sae = self._setup()
        rng = np.random.default_rng(7)
        spans = np.sort(rng.choice(len(bins) - 2, 40, replace=False)) * 128
        ccre = {"PLS": {"chr1": np.stack([spans, spans + 256], axis=1)}}
        result = C.match_concepts(sae, acts, bins, ccre, bin_bp=128,
                                  n_permutations=12, seed=0)
        row = result.per_concept.iloc[0]
        for column in ("null_p95", "sae_excess", "raw_null_p95", "raw_excess",
                       "calibrated_advantage", "sae_minus_raw"):
            assert column in result.per_concept.columns
        assert row.sae_excess == pytest.approx(row.best_auroc - row.null_p95)
        assert row.raw_excess == pytest.approx(row.raw_best_auroc - row.raw_null_p95)
        assert row.calibrated_advantage == pytest.approx(
            row.sae_excess - row.raw_excess)
        # The two ceilings differ, which is the reason calibration is needed.
        # The direction is not predictable: a wider dictionary pushes the
        # ceiling up, but TopK sparsity ties most bins at zero and pushes it
        # back down. Which wins is an empirical question per tap.
        assert abs(row.null_p95 - row.raw_null_p95) > 1e-6

    def test_dense_block_size_does_not_change_the_answer(self):
        rng = np.random.default_rng(3)
        n = 150
        labels = np.zeros(n, dtype=bool)
        labels[rng.choice(n, 40, replace=False)] = True
        acts = rng.normal(size=(n, 37)).astype(np.float32)
        whole = C.auroc_dense(acts, labels, block=1000)
        split = C.auroc_dense(acts, labels, block=7)
        assert whole == pytest.approx(split)
