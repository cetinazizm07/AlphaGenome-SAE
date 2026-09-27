"""Unit tests for ag_sae.intervene: element masking, norm dose matching, and edit mechanics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from ag_sae.intervene import (
    DiscoveredFeatures,
    _CATALOG_CACHE,
    checked_element_mask,
    cleanup_gpu,
    eligible_genes,
    match_control_norm,
    select_tracks,
)


class TestCheckedElementMask(unittest.TestCase):
    def test_valid_mask(self):
        mask = np.zeros(8192, dtype=bool)
        mask[100:105] = True
        out = checked_element_mask(mask, 8192)
        self.assertEqual(len(out), 8192)
        self.assertEqual(out.sum(), 5)

    def test_all_false_mask_raises(self):
        mask = np.zeros(8192, dtype=bool)
        with self.assertRaises(ValueError):
            checked_element_mask(mask, 8192)

    def test_wrong_length_mask_raises(self):
        mask = np.zeros(100, dtype=bool)
        mask[10] = True
        with self.assertRaises(ValueError):
            checked_element_mask(mask, 8192)


class TestDiscoveredFeatures(unittest.TestCase):
    def test_as_dict_structure(self):
        features = DiscoveredFeatures(
            promoter_feature_id=199,
            insulator_feature_id=4586,
            enhancer_feature_id=7205,
            shuffled_control_feature_id=5454,
        )
        mapping = features.as_dict()
        self.assertIn("PLS_Promoter", mapping)
        self.assertIn("CTCF_Insulator", mapping)
        self.assertIn("Enhancer_pELS", mapping)
        self.assertIn("Shuffled_Control", mapping)

        self.assertEqual(mapping["PLS_Promoter"], (199, "cCRE_PLS"))
        self.assertEqual(mapping["CTCF_Insulator"], (4586, "cCRE_CTCF_only"))


@unittest.skipIf(torch is None, "PyTorch not available in local environment")
class TestMatchControlNorm(unittest.TestCase):
    def test_dose_matching_matches_target_l2_norm(self):
        baseline = torch.zeros((1, 8192, 64), dtype=torch.float32)
        target = torch.zeros((1, 8192, 64), dtype=torch.float32)
        control = torch.zeros((1, 8192, 64), dtype=torch.float32)

        tss_bin = 4000
        target[0, tss_bin, 0] = 5.0
        control[0, tss_bin, 1] = 2.0

        matched, info = match_control_norm(baseline, target, control, tss_bin)

        target_norm = float(torch.linalg.vector_norm(target[0, tss_bin] - baseline[0, tss_bin]))
        matched_norm = float(torch.linalg.vector_norm(matched[0, tss_bin] - baseline[0, tss_bin]))

        self.assertAlmostEqual(target_norm, 5.0, places=5)
        self.assertAlmostEqual(matched_norm, 5.0, places=5)
        self.assertAlmostEqual(info["scale"], 2.5, places=5)

    def test_near_zero_target_raises_value_error(self):
        baseline = torch.zeros((1, 8192, 64), dtype=torch.float32)
        target = torch.zeros((1, 8192, 64), dtype=torch.float32)
        control = torch.zeros((1, 8192, 64), dtype=torch.float32)
        control[0, 100, 0] = 2.0

        with self.assertRaises(ValueError):
            match_control_norm(baseline, target, control, 100)


class TestCleanupGPU(unittest.TestCase):
    def test_cleanup_gpu_executes_without_error(self):
        # Should execute safely whether PyTorch or CUDA is present or not
        cleanup_gpu()


class TestTrackCatalogCache(unittest.TestCase):
    def setUp(self):
        _CATALOG_CACHE.clear()

    def tearDown(self):
        _CATALOG_CACHE.clear()

    def test_select_tracks_uses_catalog_cache(self):
        mock_track_pos = MagicMock()
        mock_track_pos.track_index = 0
        mock_track_pos.track_name = "test_pos"
        mock_track_pos.extras = {"strand": "+", "genetically_modified": False, "assay_title": "total RNA-seq"}

        mock_catalog = MagicMock()
        mock_catalog.get_tracks.return_value = [mock_track_pos]

        _CATALOG_CACHE["human"] = mock_catalog

        tracks_1 = select_tracks("+")
        self.assertIn("rna_seq", tracks_1)
        self.assertEqual(len(tracks_1["rna_seq"]), 1)
        self.assertEqual(mock_catalog.get_tracks.call_count, 3)

        # Second call should use cached tracks and not call get_tracks again
        tracks_2 = select_tracks("+")
        self.assertEqual(mock_catalog.get_tracks.call_count, 3)
        self.assertEqual(tracks_1, tracks_2)


class TestEligibleGenesCaching(unittest.TestCase):
    def test_eligible_genes_cache_hit_and_deterministic_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            gtf_path = tmp_path / "test.gtf"
            gtf_path.write_text("# mock gtf\n")

            manifest = pd.DataFrame({"split": ["test"], "chrom": ["chr1"], "win_start": [0], "win_end": [1_048_576]})
            annotations = pd.DataFrame({"split": ["test"], "chrom": ["chr1"], "bin_start": [0], "cCRE_PLS": [False]})
            cache_dir = tmp_path / "cache"

            # First run: should execute and write cache file
            result1 = eligible_genes(gtf_path, manifest, annotations, cache_dir=cache_dir)
            cache_files = list(cache_dir.glob("eligible_genes_*.json"))
            self.assertEqual(len(cache_files), 1)

            # Check contents of cache file
            cached_data = json.loads(cache_files[0].read_text())
            self.assertEqual(result1, cached_data)

            # Overwrite cache file with distinct marker data to verify cache hit
            marker_data = [{"gene_name": "CACHED_GENE", "transcript_id": "T1", "selection_hash": "abc"}]
            cache_files[0].write_text(json.dumps(marker_data))

            # Second run: should return marker data from cache without parsing GTF
            result2 = eligible_genes(gtf_path, manifest, annotations, cache_dir=cache_dir)
            self.assertEqual(result2, marker_data)


class TestContiguousWritableConversion(unittest.TestCase):
    def test_non_writable_array_converted_safely(self):
        arr = np.ones((10, 10), dtype=np.float32)
        arr.flags.writeable = False
        self.assertFalse(arr.flags.writeable)

        safe_arr = np.ascontiguousarray(arr.copy())
        self.assertTrue(safe_arr.flags.writeable)
        self.assertTrue(safe_arr.flags.c_contiguous)

        if torch is not None:
            # Must not raise UserWarning: The given NumPy array is not writable
            import warnings
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                tensor = torch.from_numpy(safe_arr)
                for w in recorded:
                    self.assertNotIn("The given NumPy array is not writable", str(w.message))
            self.assertEqual(tensor.shape, (10, 10))


if __name__ == "__main__":
    unittest.main()
