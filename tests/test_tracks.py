"""Track export: the file contract, not how a browser chooses to draw it."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from ag_sae import tracks as T


def bins(n=6, chrom="chr7", start=1000, bp=128):
    edges = start + np.arange(n) * bp
    return pd.DataFrame({"chrom": chrom, "bin_start": edges, "bin_end": edges + bp})


class TestBedGraph:
    def test_zero_bins_are_dropped(self, tmp_path):
        values = np.array([0.0, 1.5, 0.0, 2.0, 0.0, 0.0])
        path = T.write_bedgraph(bins(), values, tmp_path / "f.bedgraph", name="feature 3")
        lines = path.read_text().splitlines()
        assert lines[0].startswith("track type=bedGraph")
        assert 'name="feature 3"' in lines[0]
        assert len(lines) == 3               # header plus the two non-zero bins
        assert lines[1].split("\t") == ["chr7", "1128", "1256", "1.5"]

    def test_keeping_zeros_writes_every_bin(self, tmp_path):
        path = T.write_bedgraph(bins(), np.zeros(6), tmp_path / "f.bedgraph",
                                name="f", drop_zeros=False)
        assert len(path.read_text().splitlines()) == 7

    def test_gzip_is_chosen_by_the_suffix(self, tmp_path):
        path = T.write_bedgraph(bins(), np.ones(6), tmp_path / "f.bedgraph.gz", name="f")
        with gzip.open(path, "rt") as handle:
            assert handle.readline().startswith("track")

    def test_colour_becomes_an_igv_triple(self):
        assert T._rgb("#2a78d6") == "42,120,214"
        assert T._rgb("000000") == "0,0,0"

    def test_length_mismatch_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="values for"):
            T.write_bedgraph(bins(6), np.ones(5), tmp_path / "f", name="f")

    def test_non_finite_values_are_refused(self, tmp_path):
        values = np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="non-finite"):
            T.write_bedgraph(bins(), values, tmp_path / "f", name="f")

    def test_rows_come_out_sorted(self, tmp_path):
        frame = bins(4)
        frame = frame.iloc[::-1].reset_index(drop=True)     # hand it back to front
        path = T.write_bedgraph(frame, np.arange(1.0, 5.0), tmp_path / "f", name="f")
        starts = [int(line.split("\t")[1]) for line in path.read_text().splitlines()[1:]]
        assert starts == sorted(starts)


class TestIntervalBed:
    def test_intervals_are_written_and_sorted(self, tmp_path):
        intervals = {"chr7": np.array([[500, 600], [100, 200]], dtype=np.int64),
                     "chr1": np.array([[10, 20]], dtype=np.int64)}
        path = T.write_interval_bed(intervals, tmp_path / "pls.bed", name="PLS")
        rows = [line.split("\t")[:3] for line in path.read_text().splitlines()[1:]]
        assert rows == [["chr1", "10", "20"], ["chr7", "100", "200"], ["chr7", "500", "600"]]

    def test_an_empty_chromosome_is_skipped(self, tmp_path):
        path = T.write_interval_bed({"chr7": np.empty((0, 2))}, tmp_path / "e.bed",
                                    name="empty")
        assert len(path.read_text().splitlines()) == 1

    def test_bad_shape_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            T.write_interval_bed({"chr7": np.arange(6)}, tmp_path / "x.bed", name="x")


class TestConfig:
    def test_config_is_plain_data(self, tmp_path):
        import json
        config = T.igv_config(
            [T.bedgraph_track(tmp_path / "a.bedgraph", name="feature 1"),
             T.annotation_track(tmp_path / "p.bed", name="PLS")],
            locus="chr7:30,000,000-30,153,600")
        assert json.loads(json.dumps(config)) == config
        assert config["genome"] == "hg38"
        assert [t["type"] for t in config["tracks"]] == ["wig", "annotation"]

    def test_a_locus_is_required(self):
        with pytest.raises(ValueError, match="locus"):
            T.igv_config([], locus="")
