"""Tests for feature matching helpers."""

import numpy as np
from scipy import sparse

from ag_sae.match import (auroc_from_dense_ranks, auroc_from_sparse_ranks,
                          build_sparse_rank_struct, rank_once_dense)


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


def test_in_place_sparse_rank_deltas_match_dense_auroc():
    rng = np.random.default_rng(11)
    activations = rng.exponential(size=(401, 29)).astype(np.float32)
    activations[rng.random(activations.shape) < 0.82] = 0
    labels = rng.random((401, 6)) < np.linspace(0.08, 0.65, 6)

    expected = auroc_from_dense_ranks(rank_once_dense(activations), labels)
    delta, base, n_rows, firing_rate = build_sparse_rank_struct(
        sparse.csc_matrix(activations))
    actual = auroc_from_sparse_ranks(delta, base, n_rows, labels)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        firing_rate, np.count_nonzero(activations, axis=0) / len(activations),
        rtol=0, atol=0)
