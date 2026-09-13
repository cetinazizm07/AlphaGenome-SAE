"""Extraction: encoding, masking, hooks, shards. Hooks are the fragile part."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ag_sae import extract as E


class TestSequence:
    def test_one_hot_marks_known_bases_and_zeroes_the_rest(self):
        got = E.one_hot("ACGTNn")
        np.testing.assert_array_equal(got[:4], np.eye(4, dtype=np.float32))
        assert got[4:].sum() == 0            # N rows are all zero

    def test_lowercase_counts_as_known(self):
        np.testing.assert_array_equal(E.one_hot("acgt"), E.one_hot("ACGT"))

    def test_valid_bins_need_every_base_known(self):
        onehot = E.one_hot("AAAA" + "AANA" + "CCCC")
        np.testing.assert_array_equal(E.valid_bins(onehot, 4), [True, False, True])

    def test_valid_bins_rejects_a_ragged_length(self):
        with pytest.raises(ValueError, match="not a multiple"):
            E.valid_bins(E.one_hot("AAA"), 2)


class TestPositions:
    def test_only_valid_bins_are_ever_chosen(self):
        valid = np.zeros(100, dtype=bool)
        valid[[3, 17, 42, 88]] = True
        got = E.choose_positions(valid, 3, np.random.default_rng(0))
        assert set(got) <= {3, 17, 42, 88} and len(got) == 3
        assert (np.diff(got) > 0).all()       # sorted

    def test_asking_for_more_than_exists_returns_all(self):
        valid = np.array([True, False, True])
        np.testing.assert_array_equal(E.choose_positions(valid, 99, np.random.default_rng(0)), [0, 2])

    def test_same_seed_same_positions(self):
        valid = np.ones(1000, dtype=bool)
        a = E.choose_positions(valid, 10, np.random.default_rng(7))
        b = E.choose_positions(valid, 10, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)


# --- a stand-in with AlphaGenome's shape, so the hooks are exercised --------


class FakeEncoder(nn.Module):
    """Returns (trunk, intermediates) with intermediates in NCL, like the port."""

    def __init__(self, channels: dict[int, int]) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x):                       # x: (1, L, 4)
        length = x.shape[1]
        intermediates = {}
        for bin_bp, width in self.channels.items():
            positions = length // bin_bp
            base = x[0, ::bin_bp, :1].transpose(0, 1)          # (1, positions)
            intermediates[f"bin_size_{bin_bp}"] = (
                base.expand(width, positions).unsqueeze(0) + bin_bp)   # (1, C, L)
        trunk = torch.zeros(1, length // 128, 64)
        return trunk, intermediates


class FakeMHA(nn.Module):
    def forward(self, x, bias=None):
        return x * 0.0


class FakeTower(nn.Module):
    def __init__(self, blocks: int, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.ModuleDict({"mha": FakeMHA(), "mlp": nn.Identity()}) for _ in range(blocks))

    def _forward_block(self, block, x):
        # Mirrors the real tower: the adds live here, not inside a module.
        x = x + block["mha"](x)
        return x + block["mlp"](x)

    def forward(self, x):
        for block in self.blocks:
            x = self._forward_block(block, x)
        return x


class FakeModel(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.encoder = FakeEncoder({4: 3, 128: width})
        self.tower = FakeTower(9, width)
        self.width = width

    def forward(self, onehot: np.ndarray):
        x = torch.from_numpy(onehot).unsqueeze(0)
        _, intermediates = self.encoder(x)
        trunk = intermediates["bin_size_128"][0].T.unsqueeze(0)   # (1, L, C)
        self.tower(trunk)
        return None


def fake_taps():
    return [E.Tap("bin_size_4", "encoder", "bin_size_4", 4, 3),
            E.Tap("resid_pre_b0", "tower", 0, 128, 6),
            E.Tap("resid_pre_b4", "tower", 4, 128, 6)]


class TestCapture:
    def test_pre_hooks_capture_the_residual_stream(self):
        model, taps = FakeModel(), fake_taps()
        onehot = E.one_hot("ACGT" * 64)
        with E.Capture(model, taps) as capture:
            model(onehot)
            out = capture.outputs()
        assert set(out) == {t.name for t in taps}
        assert out["bin_size_4"].shape == (64, 3)
        assert out["resid_pre_b0"].shape == (2, 6)

    def test_a_plain_forward_hook_on_the_block_would_never_fire(self):
        """The reason pre-hooks are used: ModuleDict has no forward."""
        model = FakeModel()
        fired = []
        model.tower.blocks[0].register_forward_hook(lambda *a: fired.append(1))
        model(E.one_hot("ACGT" * 64))
        assert fired == []

    def test_missing_hook_output_is_an_error_not_silence(self):
        model = FakeModel()
        with E.Capture(model, fake_taps()) as capture:
            with pytest.raises(RuntimeError, match="did not fire"):
                capture.outputs()

    def test_handles_are_removed_on_exit(self):
        model = FakeModel()
        with E.Capture(model, fake_taps()):
            pass
        fired = []
        model.tower.blocks[0]["mha"].register_forward_pre_hook(
            lambda *a: fired.append(1))
        model(E.one_hot("ACGT" * 64))
        assert len(fired) == 1          # only the new hook, not a leaked one


# --- shards ----------------------------------------------------------------


def tiny_manifest(n: int = 4, window: int = 2048) -> pd.DataFrame:
    starts = np.arange(n) * window
    return pd.DataFrame({"chrom": "chr1", "win_start": starts, "win_end": starts + window,
                         "split": ["train"] * (n - 1) + ["val"]})


def run(tmp_path, manifest, keep=8, **kwargs):
    model, taps = FakeModel(), fake_taps()
    rng = np.random.default_rng(0)

    def sequence_for(chrom, start, end):
        return "".join(rng.choice(list("ACGT"), end - start))

    def forward(onehot):
        with E.Capture(model, taps) as capture:
            model(onehot)
            return capture.outputs()

    return E.run_extraction(manifest, sequence_for, forward, taps, tmp_path,
                            positions_per_window=keep, windows_per_shard=2,
                            log=lambda *a: None, **kwargs)


class TestExtraction:
    def test_writes_one_shard_set_per_tap_and_reads_back(self, tmp_path):
        index = run(tmp_path, tiny_manifest())
        assert index["complete"]
        store = E.ShardStore(tmp_path, "bin_size_4", "train")
        assert store.dim == 3 and len(store) == 3 * 8      # 3 train windows x 8 kept
        rows = store.take(np.arange(len(store)))
        assert rows.dtype == np.float32 and np.isfinite(rows).all()
        coords = store.coordinates()
        assert len(coords) == len(store)
        assert (coords.bin_end - coords.bin_start == 4).all()

    def test_taps_keep_the_same_number_of_positions_at_different_widths(self, tmp_path):
        run(tmp_path, tiny_manifest())
        fine = E.ShardStore(tmp_path, "bin_size_4", "train")
        coarse = E.ShardStore(tmp_path, "resid_pre_b0", "train")
        assert len(fine) == len(coarse)          # matched rows, not matched windows
        assert fine.dim == 3 and coarse.dim == 6

    def test_the_coarsest_tap_caps_positions_per_window(self, tmp_path):
        """A 512 bp window holds only 4 bins at 128 bp, so keep=8 cannot be met."""
        run(tmp_path, tiny_manifest(window=512), keep=8)
        assert len(E.ShardStore(tmp_path, "bin_size_4", "train")) == 3 * 8
        assert len(E.ShardStore(tmp_path, "resid_pre_b0", "train")) == 3 * 4

    def test_rerunning_is_a_no_op(self, tmp_path):
        first = run(tmp_path, tiny_manifest())
        again = run(tmp_path, tiny_manifest())
        assert again["shards"] == first["shards"]

    def test_a_different_setting_refuses_to_share_the_directory(self, tmp_path):
        run(tmp_path, tiny_manifest())
        with pytest.raises(ValueError, match="different extraction"):
            run(tmp_path, tiny_manifest(), keep=4)

    def test_split_is_not_mixed_inside_a_shard(self, tmp_path):
        index = run(tmp_path, tiny_manifest())
        assert {r["split"] for r in index["shards"]} == {"train", "val"}
        for record in index["shards"]:
            assert record["split"] in ("train", "val")

    def test_store_order_covers_every_row_once(self, tmp_path):
        run(tmp_path, tiny_manifest())
        store = E.ShardStore(tmp_path, "bin_size_4", "train")
        order = store.order(seed=1, epoch=0)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(store)))
        assert not np.array_equal(order, store.order(seed=1, epoch=1))

    def test_unknown_tap_or_split_is_an_error(self, tmp_path):
        run(tmp_path, tiny_manifest())
        with pytest.raises(ValueError, match="no rows"):
            E.ShardStore(tmp_path, "bin_size_4", "test")


# --- FASTA random access ---------------------------------------------------


class TestFastaIndex:
    def write(self, tmp_path, records, line_bases=6):
        path = tmp_path / "g.fa"
        with open(path, "w") as handle:
            for name, sequence in records:
                handle.write(f">{name} some description\n")
                for i in range(0, len(sequence), line_bases):
                    handle.write(sequence[i:i + line_bases] + "\n")
        return path

    def test_fetch_matches_the_original_sequence(self, tmp_path):
        rng = np.random.default_rng(0)
        seq1 = "".join(rng.choice(list("ACGTN"), 250))
        seq2 = "".join(rng.choice(list("ACGT"), 97))
        index = E.FastaIndex(self.write(tmp_path, [("chr1", seq1), ("chr2", seq2)]))
        assert index.length("chr1") == 250 and index.length("chr2") == 97
        for start, end in [(0, 10), (5, 6), (0, 250), (123, 200), (249, 250)]:
            assert index.fetch("chr1", start, end) == seq1[start:end], (start, end)
        assert index.fetch("chr2", 10, 40) == seq2[10:40]

    def test_reading_past_the_end_pads_with_n(self, tmp_path):
        index = E.FastaIndex(self.write(tmp_path, [("chr1", "ACGTACGTAC")]))
        assert index.fetch("chr1", 6, 16) == "GTAC" + "N" * 6   # seq[6:10] then padding
        assert index.fetch("chr1", 20, 25) == "NNNNN"

    def test_unknown_chromosome_and_bad_range(self, tmp_path):
        index = E.FastaIndex(self.write(tmp_path, [("chr1", "ACGT" * 5)]))
        with pytest.raises(KeyError, match="chrZ"):
            index.fetch("chrZ", 0, 4)
        with pytest.raises(ValueError, match="bad range"):
            index.fetch("chr1", 5, 5)

    def test_gzipped_input_is_refused_with_a_clear_message(self, tmp_path):
        path = tmp_path / "g.fa.gz"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="uncompress"):
            E.FastaIndex(path)

    def test_chrom_sizes_helper(self, tmp_path):
        index = E.FastaIndex(self.write(tmp_path, [("chr1", "A" * 30), ("chr2", "C" * 12)]))
        assert E.chrom_sizes(index) == {"chr1": 30, "chr2": 12}

    def test_name_stops_at_the_first_space(self, tmp_path):
        index = E.FastaIndex(self.write(tmp_path, [("chr1", "ACGT" * 4)]))
        assert list(index.records) == ["chr1"]
