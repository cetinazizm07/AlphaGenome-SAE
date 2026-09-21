"""Complementary, dependency-free metrics for genomic concept recovery."""

import numpy as np


def average_precision(labels, scores):
    """Non-interpolated average precision with exact handling of score ties.

    This is the area under the step-wise precision-recall curve.  Its random
    baseline is the positive prevalence, unlike AUROC's fixed 0.5 baseline.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    n_positive = int(labels.sum())
    if n_positive == 0 or n_positive == len(labels):
        raise ValueError("average precision requires both classes")

    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_labels = labels[order]
    # Evaluate only after the last member of each tied score group.  Splitting
    # ties according to row order would make AP depend on an arbitrary ordering.
    group_end = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    tp = np.cumsum(ranked_labels, dtype=np.int64)[group_end]
    seen = np.arange(1, len(labels) + 1, dtype=np.int64)[group_end]
    recall = tp / n_positive
    precision = tp / seen
    recall_before = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_before) * precision))


def domain_precision_recall_f1(predicted, labels, chrom, starts, bin_bp=128):
    """Paper-style bin precision and contiguous-domain recall.

    A positive domain is recovered when at least one predicted-positive bin
    overlaps it.  Chromosome changes and coordinate gaps split domains, so
    masked or unobserved genomic territory is never treated as contiguous.
    """
    predicted = np.asarray(predicted, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    chrom = np.asarray(chrom)
    starts = np.asarray(starts)
    n = len(labels)
    if any(len(x) != n for x in (predicted, chrom, starts)) or n == 0:
        raise ValueError("domain metric inputs must be nonempty and aligned")
    if bin_bp < 1:
        raise ValueError("bin_bp must be positive")

    n_predicted = int(predicted.sum())
    precision = (float(np.count_nonzero(predicted & labels)) / n_predicted
                 if n_predicted else 0.0)

    positive_rows = np.flatnonzero(labels)
    if not len(positive_rows):
        raise ValueError("domain recall requires at least one positive domain")
    breaks = np.r_[True,
                   (chrom[positive_rows[1:]] != chrom[positive_rows[:-1]]) |
                   (starts[positive_rows[1:]] != starts[positive_rows[:-1]] + bin_bp)]
    domain_ids = np.cumsum(breaks) - 1
    n_domains = int(domain_ids[-1] + 1)
    recovered = np.zeros(n_domains, dtype=bool)
    np.logical_or.at(recovered, domain_ids, predicted[positive_rows])
    recall = float(recovered.mean())
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_domains": n_domains, "n_domains_recovered": int(recovered.sum())}


def genomic_block_ids(chrom, starts, block_bp=131072):
    """Return stable integer IDs for fixed genomic bootstrap blocks."""
    chrom = np.asarray(chrom)
    starts = np.asarray(starts)
    if chrom.ndim != 1 or starts.ndim != 1 or len(chrom) != len(starts) or not len(starts):
        raise ValueError("chrom and starts must be nonempty aligned vectors")
    if block_bp < 1 or starts.dtype.kind not in "iu" or (starts < 0).any():
        raise ValueError("block_bp and genomic starts must be nonnegative integers")
    keys = np.asarray([f"{c}:{int(s) // block_bp}" for c, s in zip(chrom, starts)])
    _, ids = np.unique(keys, return_inverse=True)
    return ids.astype(np.int64)


def _event_block_summaries(scores, labels, chrom, starts, block_ids,
                           radius_bins=2, sigma_bins=1.0, bin_bp=128):
    """Aggregate Gaussian event signal and background signal by genomic block."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    chrom = np.asarray(chrom)
    starts = np.asarray(starts)
    block_ids = np.asarray(block_ids)
    n = len(scores)
    if any(x.ndim != 1 or len(x) != n for x in (labels, chrom, starts, block_ids)) or n == 0:
        raise ValueError("event metric inputs must be nonempty aligned vectors")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("event enrichment requires finite nonnegative activations")
    if starts.dtype.kind not in "iu" or (starts < 0).any() or bin_bp < 1:
        raise ValueError("event coordinates must be nonnegative integer bin starts")
    if not isinstance(radius_bins, (int, np.integer)) or radius_bins < 0:
        raise ValueError("radius_bins must be a nonnegative integer")
    if not np.isfinite(sigma_bins) or sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    if not labels.any():
        raise ValueError("event enrichment requires at least one event")

    # Coordinate sorting makes the statistic invariant to manifest row order.
    order = np.lexsort((starts, chrom.astype(str)))
    scores = scores[order]
    labels = labels[order]
    chrom = chrom[order]
    starts = starts[order]
    block_ids = block_ids[order]
    if np.any((chrom[1:] == chrom[:-1]) & (starts[1:] == starts[:-1])):
        raise ValueError("event coordinates must be unique")

    contiguous = ((chrom[1:] == chrom[:-1]) &
                  (starts[1:] == starts[:-1] + bin_bp))
    run_start = np.r_[0, np.flatnonzero(~contiguous) + 1]
    run_end = np.r_[run_start[1:], n]
    run_of = np.empty(n, dtype=np.int64)
    for run, (lo, hi) in enumerate(zip(run_start, run_end)):
        run_of[lo:hi] = run

    # One event per contiguous positive domain. At 128-bp resolution multiple
    # base-pair events inside the same labelled domain cannot be distinguished.
    positive = np.flatnonzero(labels)
    domain_break = np.r_[True,
                         (run_of[positive[1:]] != run_of[positive[:-1]]) |
                         (positive[1:] != positive[:-1] + 1)]
    domain_start_pos = np.flatnonzero(domain_break)
    domain_end_pos = np.r_[domain_start_pos[1:], len(positive)]
    centers = np.array([
        positive[(lo + hi - 1) // 2] for lo, hi in zip(domain_start_pos, domain_end_pos)
    ], dtype=np.int64)

    event_values = np.empty(len(centers), dtype=np.float64)
    event_blocks = np.empty(len(centers), dtype=block_ids.dtype)
    excluded = np.zeros(n, dtype=bool)
    for j, center in enumerate(centers):
        run = run_of[center]
        lo = max(int(run_start[run]), center - radius_bins)
        hi = min(int(run_end[run]), center + radius_bins + 1)
        local_offsets = np.arange(lo - center, hi - center)
        weights = np.exp(-(local_offsets.astype(np.float64) ** 2) /
                         (2.0 * sigma_bins ** 2))
        event_values[j] = float(np.dot(weights, scores[lo:hi]) / weights.sum())
        event_blocks[j] = block_ids[center]
        excluded[lo:hi] = True

    # Never let a labelled event bin enter the background, including the edge
    # of a multi-bin domain that extends beyond the center-based kernel.
    excluded |= labels

    background = ~excluded
    if not background.any():
        raise ValueError("event windows leave no background bins")
    unique_blocks, normalized_blocks = np.unique(block_ids, return_inverse=True)
    block_lookup = {value: i for i, value in enumerate(unique_blocks)}
    event_block_index = np.asarray([block_lookup[value] for value in event_blocks])
    n_blocks = len(unique_blocks)
    event_sum = np.bincount(event_block_index, weights=event_values, minlength=n_blocks)
    event_count = np.bincount(event_block_index, minlength=n_blocks).astype(np.int64)
    background_sum = np.bincount(
        normalized_blocks[background], weights=scores[background], minlength=n_blocks)
    background_count = np.bincount(
        normalized_blocks[background], minlength=n_blocks).astype(np.int64)
    return event_sum, event_count, background_sum, background_count


def event_enrichment_with_block_bootstrap(
        scores, labels, chrom, starts, block_ids, radius_bins=2,
        sigma_bins=1.0, n_bootstrap=2000, seed=0, confidence=0.95,
        bin_bp=128, epsilon=1e-12):
    """Gaussian sparse-event enrichment with a genomic block-bootstrap CI.

    Blocks are sampled with replacement. Event and background means are then
    reconstructed from block-level sums/counts, preserving all within-window
    spatial dependence. The interval is conditional on the already selected
    feature; feature-selection multiplicity is handled separately by the
    maximum-feature permutation null in ``match``.
    """
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0 < confidence < 1 or epsilon <= 0:
        raise ValueError("confidence and epsilon must be positive and valid")
    event_sum, event_count, background_sum, background_count = _event_block_summaries(
        scores, labels, chrom, starts, block_ids, radius_bins, sigma_bins, bin_bp)

    total_events = int(event_count.sum())
    total_background = int(background_count.sum())
    event_mean = float(event_sum.sum() / total_events)
    background_mean = float(background_sum.sum() / total_background)
    estimate = float(np.log((event_mean + epsilon) / (background_mean + epsilon)))

    rng = np.random.default_rng(seed)
    n_blocks = len(event_sum)
    draws = rng.integers(0, n_blocks, size=(n_bootstrap, n_blocks))
    boot_event_count = event_count[draws].sum(axis=1)
    boot_background_count = background_count[draws].sum(axis=1)
    valid = (boot_event_count > 0) & (boot_background_count > 0)
    values = np.full(n_bootstrap, np.nan, dtype=np.float64)
    if valid.any():
        boot_event_mean = event_sum[draws].sum(axis=1)[valid] / boot_event_count[valid]
        boot_background_mean = (background_sum[draws].sum(axis=1)[valid] /
                                boot_background_count[valid])
        values[valid] = np.log((boot_event_mean + epsilon) /
                               (boot_background_mean + epsilon))
        alpha = (1.0 - confidence) / 2.0
        ci_low, ci_high = np.nanquantile(values, [alpha, 1.0 - alpha])
    else:
        ci_low = ci_high = np.nan
    return {
        "log_enrichment": estimate,
        "event_mean": event_mean,
        "background_mean": background_mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_events": total_events,
        "n_background_bins": total_background,
        "n_blocks": n_blocks,
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_valid": int(valid.sum()),
    }
