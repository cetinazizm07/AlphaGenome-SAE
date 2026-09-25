import numpy as np
import pandas as pd
import pytest

from ag_sae import extract
from ag_sae.data import MatchShardStore, sha256


def _fake_native_store(monkeypatch, root, coordinates, values):
    act_path = root / "activations.bin"
    coord_path = root / "coordinates.bin"
    act_path.write_bytes(b"test activation shard")
    coord_path.write_bytes(b"test coordinate shard")

    class FakeStore:
        def __init__(self, directory, tap, split):
            assert tap == "resid_pre_b0" and split == "test"
            self.dim = values.shape[1]
            self._coordinates = coordinates.reset_index(drop=True)
            self._values = values.astype(np.float32)
            self.records = [{
                "activations": act_path.name,
                "coordinates": coord_path.name,
                "activations_sha256": sha256(act_path),
                "coordinates_sha256": sha256(coord_path),
            }]
            self.identity = {"index": "test-index", "tap": tap, "split": split}

        def coordinates(self):
            return self._coordinates.copy()

        def take(self, indices):
            return self._values[np.asarray(indices, dtype=np.int64)]

    monkeypatch.setattr(extract, "ShardStore", FakeStore)


def test_native_shards_match_annotations_by_coordinates_and_mask(tmp_path, monkeypatch):
    ann = pd.DataFrame({
        "chrom": ["chr1"] * 4,
        "bin_start": np.array([0, 128, 256, 384], dtype=np.int64),
        "bin_end": np.array([128, 256, 384, 512], dtype=np.int64),
        "split": ["test"] * 4,
        "n_mask": [True, True, False, True],
        "concept": [False, True, True, False],
    })
    coords = ann.iloc[[1, 0, 2]].drop(columns=["n_mask", "concept"])
    values = np.array([[20, 21], [10, 11], [30, 31]], dtype=np.float16)
    _fake_native_store(monkeypatch, tmp_path, coords, values)

    # A non-RangeIndex must not be mistaken for row positions when labels are
    # subsequently selected with iloc in the matching pipeline.
    ann.index = [20, 10, 30, 40]
    store = MatchShardStore(tmp_path, "resid_pre_b0", "test", ann)

    np.testing.assert_array_equal(store.annotation_indices, [1, 0])
    np.testing.assert_array_equal(store.take([0, 1]), values[:2].astype(np.float32))
    assert store.coordinates().bin_start.tolist() == [128, 0]
    assert ann.iloc[store.annotation_indices].concept.tolist() == [True, False]


def test_native_matching_rejects_non_128bp_taps(tmp_path):
    ann = pd.DataFrame({
        "chrom": ["chr1"], "bin_start": np.array([0], dtype=np.int64),
        "bin_end": np.array([128], dtype=np.int64), "split": ["test"],
        "n_mask": [True],
    })
    with pytest.raises(ValueError, match="128-bp tower tap"):
        MatchShardStore(tmp_path, "bin_size_64", "test", ann)
