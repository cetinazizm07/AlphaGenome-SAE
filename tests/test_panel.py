"""Concept derivations. Strand handling and label parsing are the risky parts."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from ag_sae import panel as P


def bed_gz(tmp_path, rows, name="ccre.bed.gz"):
    path = tmp_path / name
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join(str(x) for x in row) + "\n")
    return path


def gtf(tmp_path, rows, name="anno.gtf"):
    path = tmp_path / name
    with open(path, "w") as handle:
        handle.write("##description: test\n")
        for chrom, feature, start, end, strand, transcript in rows:
            handle.write(f'{chrom}\tTEST\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t'
                         f'gene_id "G"; transcript_id "{transcript}";\n')
    return path


class TestCCRE:
    def rows(self):
        # Column 10 (index 9) holds comma-separated labels, as in ENCFF234XEZ.
        return [("chr1", 100, 300, "E1", 0, ".", 100, 300, "0,0,0", "PLS"),
                ("chr1", 500, 700, "E2", 0, ".", 500, 700, "0,0,0", "pELS,CTCF-bound"),
                ("chr1", 900, 950, "E3", 0, ".", 900, 950, "0,0,0", "CTCF-only,CTCF-bound"),
                ("chr2", 100, 200, "E4", 0, ".", 100, 200, "0,0,0", "dELS")]

    def test_label_is_matched_per_item_not_by_string_equality(self, tmp_path):
        path = bed_gz(tmp_path, self.rows())
        got = P.ccre_intervals(path, "pELS", 9)
        assert [tuple(x) for x in got["chr1"]] == [(500, 700)]
        bound = P.ccre_intervals(path, "CTCF-bound", 9)
        assert len(bound["chr1"]) == 2          # pELS and CTCF-only both carry it

    def test_union_keeps_every_element(self, tmp_path):
        got = P.ccre_intervals(bed_gz(tmp_path, self.rows()), None, 9)
        assert len(got["chr1"]) == 3 and len(got["chr2"]) == 1

    def test_overlapping_elements_merge(self, tmp_path):
        rows = [("chr1", 100, 300, "A", 0, ".", 100, 300, "0,0,0", "PLS"),
                ("chr1", 250, 400, "B", 0, ".", 250, 400, "0,0,0", "PLS")]
        got = P.ccre_intervals(bed_gz(tmp_path, rows), "PLS", 9)
        assert [tuple(x) for x in got["chr1"]] == [(100, 400)]

    def test_missing_label_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="no intervals"):
            P.ccre_intervals(bed_gz(tmp_path, self.rows()), "nonsense", 9)

    def test_wrong_class_column_is_caught(self, tmp_path):
        with pytest.raises(ValueError, match="fewer than"):
            P.ccre_intervals(bed_gz(tmp_path, self.rows()), "PLS", 40)


class TestSpliceSites:
    def transcripts(self, tmp_path):
        # GTF is 1-based inclusive, so exon 101..200 is BED [100, 200).
        return gtf(tmp_path, [
            ("chr1", "exon", 101, 200, "+", "T_plus"),
            ("chr1", "exon", 301, 400, "+", "T_plus"),
            ("chr2", "exon", 101, 200, "-", "T_minus"),
            ("chr2", "exon", 301, 400, "-", "T_minus"),
        ])

    def test_donor_and_acceptor_swap_with_strand(self, tmp_path):
        path = self.transcripts(tmp_path)
        # Intron is [200, 300) for both transcripts.
        donor = P.splice_sites(path, "donor", 10)
        acceptor = P.splice_sites(path, "acceptor", 10)
        assert [tuple(x) for x in donor["chr1"]] == [(190, 210)]    # + : intron start
        assert [tuple(x) for x in donor["chr2"]] == [(290, 310)]    # - : intron end
        assert [tuple(x) for x in acceptor["chr1"]] == [(290, 310)]
        assert [tuple(x) for x in acceptor["chr2"]] == [(190, 210)]

    def test_gaps_between_different_transcripts_are_not_introns(self, tmp_path):
        path = gtf(tmp_path, [("chr1", "exon", 101, 200, "+", "A"),
                              ("chr1", "exon", 301, 400, "+", "B")])
        with pytest.raises(ValueError, match="no introns"):
            P.splice_sites(path, "donor", 10)

    def test_flank_sets_the_interval_width(self, tmp_path):
        got = P.splice_sites(self.transcripts(tmp_path), "donor", 3)
        assert [tuple(x) for x in got["chr1"]] == [(197, 203)]

    def test_rejects_an_unknown_site(self, tmp_path):
        with pytest.raises(ValueError, match="donor"):
            P.splice_sites(self.transcripts(tmp_path), "middle", 10)


class TestTSS:
    def test_start_depends_on_strand(self, tmp_path):
        path = gtf(tmp_path, [("chr1", "transcript", 1001, 2000, "+", "A"),
                              ("chr2", "transcript", 1001, 2000, "-", "B")])
        got = P.tss_windows(path, 100)
        # + : TSS at BED 1000. - : TSS at BED 1999.
        assert [tuple(x) for x in got["chr1"]] == [(900, 1101)]
        assert [tuple(x) for x in got["chr2"]] == [(1899, 2100)]

    def test_window_is_clipped_at_the_chromosome_start(self, tmp_path):
        path = gtf(tmp_path, [("chr1", "transcript", 11, 500, "+", "A")])
        assert P.tss_windows(path, 1000)["chr1"][0][0] == 0


class TestGC:
    def fasta(self, tmp_path, sequence, chrom="chr1"):
        path = tmp_path / "g.fa"
        path.write_text(f">{chrom}\n{sequence}\n")
        return path

    def test_picks_the_gc_richest_bins(self, tmp_path):
        # Four bins of 4 bp with GC fractions 0, 0.5, 1.0, 1.0.
        path = self.fasta(tmp_path, "AAAA" + "ACGT" + "GCGC" + "CGCG")
        got = P.gc_quantile_bins(path, 4, 0.75, min_bins=2)
        assert [tuple(x) for x in got["chr1"]] == [(8, 16)]    # the two GC bins, merged

    def test_bins_with_ambiguous_bases_are_excluded(self, tmp_path):
        path = self.fasta(tmp_path, "GCGC" + "NNNN" + "GCGC" + "AAAA" +
                          "ATAT" + "ATAT" + "ATAT" + "ATAT" + "ATAT" + "ATAT" + "ATAT")
        got = P.gc_quantile_bins(path, 4, 0.5, min_bins=2)
        for chrom, intervals in got.items():
            assert not ((intervals[:, 0] < 8) & (intervals[:, 1] > 4)).any()

    def test_rejects_an_out_of_range_quantile(self, tmp_path):
        with pytest.raises(ValueError, match="quantile"):
            P.gc_quantile_bins(self.fasta(tmp_path, "ACGT" * 20), 4, 1.5, min_bins=2)


class TestShuffle:
    def source(self):
        return {"chr1": np.array([[100, 200], [500, 700], [900, 1000]], dtype=np.int64)}

    def test_keeps_count_and_lengths_but_moves_them(self):
        got = P.shuffle_intervals(self.source(), {"chr1": 100_000}, seed=0)
        lengths = sorted((got["chr1"][:, 1] - got["chr1"][:, 0]).tolist())
        assert lengths == [100, 100, 200]
        assert not np.array_equal(got["chr1"], self.source()["chr1"])

    def test_avoids_excluded_regions(self):
        blocked = {"chr1": np.array([[0, 90_000]], dtype=np.int64)}
        got = P.shuffle_intervals(self.source(), {"chr1": 100_000}, seed=1, exclude=blocked)
        assert (got["chr1"][:, 0] >= 90_000).all()

    def test_same_seed_same_placement(self):
        a = P.shuffle_intervals(self.source(), {"chr1": 100_000}, seed=5)
        b = P.shuffle_intervals(self.source(), {"chr1": 100_000}, seed=5)
        np.testing.assert_array_equal(a["chr1"], b["chr1"])

    def test_missing_chromosome_length_is_an_error(self):
        with pytest.raises(KeyError, match="chr1"):
            P.shuffle_intervals(self.source(), {}, seed=0)

    def test_gives_up_rather_than_looping_forever(self):
        blocked = {"chr1": np.array([[0, 1_000_000]], dtype=np.int64)}
        with pytest.raises(RuntimeError, match="could not place"):
            P.shuffle_intervals(self.source(), {"chr1": 1000}, seed=0, exclude=blocked)


class TestPanelFile:
    def test_the_frozen_panel_is_well_formed(self):
        panel = json.loads((__import__("pathlib").Path(__file__).parents[1]
                            / "concept_panel.json").read_text())
        assert panel["schema"] == "ag-sae/concept-panel/1"
        names = [c["name"] for c in panel["concepts"]]
        assert len(names) == len(set(names))
        roles = {c["role"] for c in panel["concepts"]}
        # Every role the design relies on must be represented.
        assert {"primary", "local", "contextual",
                "positive_control", "negative_control"} <= roles
        for concept in panel["concepts"]:
            how = concept["derivation"]
            assert how["kind"] in {"ccre_class", "ccre_union", "gencode_splice",
                                   "gencode_tss_window", "gc_quantile", "shuffle_of"}
            if "source" in how:
                assert how["source"] in panel["sources"]
            if how["kind"] == "shuffle_of":
                assert how["concept"] in names

    def test_md5_mismatch_is_refused(self, tmp_path):
        panel = {"sources": {"x": {"md5": "0" * 32}}}
        path = tmp_path / "f"
        path.write_text("content")
        with pytest.raises(ValueError, match="md5 mismatch"):
            P.verify_sources(panel, {"x": path})
        assert "mismatch" in P.verify_sources(panel, {"x": path}, strict=False)["x"]
