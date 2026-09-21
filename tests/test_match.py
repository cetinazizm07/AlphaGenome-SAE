"""Tests for feature matching helpers."""

import numpy as np

from ag_sae.match import auroc_from_dense_ranks, rank_once_dense


def test_dense_chunk_sizes_do_not_change_ranks_or_auroc():
    """Memory-bounded chunks must preserve the exact dense statistics."""
    rng = np.random.default_rng(7)
    activations = rng.normal(size=(1003, 73)).astype(np.float32)
    activations[::9] = 0
    labels = rng.random((1003, 8)) < np.linspace(0.05, 0.70, 8)

    one_block = rank_once_dense(activations, feat_chunk=73)
    small_blocks = rank_once_dense(activations, feat_chunk=7)
    np.testing.assert_array_equal(small_blocks, one_block)

    expected = auroc_from_dense_ranks(one_block, labels, chunk=73)
    actual = auroc_from_dense_ranks(small_blocks, labels, chunk=5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
