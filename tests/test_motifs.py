"""Motif matching: parsing, alignment, strand handling and the empirical null."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ag_sae import motifs as M

JASPAR = """>MA0139.1 CTCF
 87  167  281   56    8
291  145   49  800  903
 76  414  449   21    0
459  187  134  036   89
>MA0060.1 NFYA
  0    0    0
  0    0  100
100    0    0
  0  100    0
"""


def sharp(pattern: str) -> np.ndarray:
    """A near-deterministic PWM spelling `pattern`."""
    out = np.full((len(pattern), 4), 0.01)
    for position, base in enumerate(pattern):
        out[position, "ACGT".index(base)] = 0.97
    return out / out.sum(axis=1, keepdims=True)


class TestJaspar:
    def test_two_motifs_are_read_with_their_names(self, tmp_path):
        path = tmp_path / "db.pfm"
        path.write_text(JASPAR)
        db = M.read_jaspar(path)
        assert sorted(db) == ["CTCF", "NFYA"]
        assert db["CTCF"].shape == (5, 4)
        assert db["NFYA"].shape == (3, 4)

    def test_columns_are_probabilities(self, tmp_path):
        path = tmp_path / "db.pfm"
        path.write_text(JASPAR)
        for matrix in M.read_jaspar(path).values():
            assert matrix.sum(axis=1) == pytest.approx(np.ones(len(matrix)))

    def test_an_empty_file_is_refused(self, tmp_path):
        path = tmp_path / "empty.pfm"
        path.write_text("\n\n")
        with pytest.raises(ValueError, match="no motifs"):
            M.read_jaspar(path)

    def test_a_truncated_matrix_is_refused(self, tmp_path):
        path = tmp_path / "bad.pfm"
        path.write_text(">MA0001.1 X\n1 2\n3 4\n")
        with pytest.raises(ValueError, match="four rows"):
            M.read_jaspar(path)


class TestPwm:
    def test_one_repeated_sequence_gives_that_sequence(self):
        pwm = M.pwm_from_sequences(["ACGT"] * 50)
        assert "".join("ACGT"[i] for i in pwm.argmax(axis=1)) == "ACGT"

    def test_weights_decide_the_consensus(self):
        pwm = M.pwm_from_sequences(["AAAA", "TAAA"], weights=np.array([0.01, 50.0]))
        assert "ACGT"[pwm.argmax(axis=1)[0]] == "T"

    def test_no_column_is_ever_zero(self):
        # A pure column must still be matchable, or no database motif can align.
        pwm = M.pwm_from_sequences(["AAAA"] * 20)
        assert (pwm > 0).all()

    def test_ragged_input_is_refused(self):
        with pytest.raises(ValueError, match="same length"):
            M.pwm_from_sequences(["AC", "ACG"])


class TestAlignment:
    def test_a_motif_matches_itself_perfectly(self):
        pwm = sharp("TGACTCA")
        score, offset = M.align_score(pwm, pwm)
        assert score == pytest.approx(1.0)
        assert offset == 0

    def test_an_offset_copy_is_found(self):
        query = sharp("NNTGACTCA".replace("N", "A"))
        target = sharp("TGACTCA")
        score, offset = M.align_score(query, target)
        assert score > 0.95
        assert offset == 2

    def test_the_reverse_strand_is_reported(self):
        target = sharp("TGACTCA")
        query = M.reverse_complement(target)
        score, strand = M.compare(query, target)
        assert score > 0.95
        assert strand == "-"

    def test_reverse_complement_is_its_own_inverse(self):
        pwm = sharp("ACGTTG")
        assert M.reverse_complement(M.reverse_complement(pwm)) == pytest.approx(pwm)

    def test_unrelated_motifs_score_low(self):
        score, _ = M.compare(sharp("AAAAAAA"), sharp("CGCGCGC"))
        assert score < 0.5


class TestMatching:
    def _db(self):
        return {"AP1": sharp("TGACTCA"), "CTCF": sharp("CCGCGAGGTGGCAG"),
                "POLY_A": sharp("AAAAAAAA"), "GC_BOX": sharp("GGGGCGGGGC")}

    def test_the_planted_motif_comes_first_and_clears_the_null(self):
        query = sharp("TGACTCA")
        out = M.match_motifs(query, self._db(), n_null=100, seed=0)
        assert out.motif.iloc[0] == "AP1"
        assert out.p_value.iloc[0] < 0.05
        assert out.score.iloc[0] > out.null_p95.iloc[0]

    def test_a_shuffled_query_matches_nothing_convincingly(self):
        rng = np.random.default_rng(3)
        query = sharp("TGACTCA")[rng.permutation(7)]
        out = M.match_motifs(query, self._db(), n_null=100, seed=0)
        assert out.p_value.min() > 0.05

    def test_a_p_value_is_never_exactly_zero(self):
        out = M.match_motifs(sharp("TGACTCA"), self._db(), n_null=20, seed=0)
        assert (out.p_value > 0).all()

    def test_top_n_limits_the_report(self):
        out = M.match_motifs(sharp("TGACTCA"), self._db(), n_null=20, top_n=2)
        assert len(out) == 2

    def test_an_empty_database_is_refused(self):
        with pytest.raises(ValueError, match="Empty motif database"):
            M.match_motifs(sharp("ACGT"), {}, n_null=5)
