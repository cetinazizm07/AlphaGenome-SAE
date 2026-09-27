"""Robust statistical hypothesis testing for paired causal intervention readouts.

Provides parametric (one-sample t-test) and non-parametric (Wilcoxon signed-rank test)
evaluations against the null hypothesis H0: Delta = 0, with robust safeguards against
zero-difference degenerate distributions (e.g. inactive sparse latents).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticalSummary:
    """Encapsulates paired parametric and non-parametric hypothesis test results."""

    mean_delta: float
    median_delta: float
    t_statistic: float
    t_p_value: float
    wilcoxon_w: float
    wilcoxon_p_value: float
    pct_negative: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON statistical reports."""
        return {
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "t_stat": self.t_statistic,
            "t_p": self.t_p_value,
            "wilcoxon_w": self.wilcoxon_w,
            "wilcoxon_p": self.wilcoxon_p_value,
            "pct_negative": self.pct_negative,
        }


def safe_wilcoxon(deltas: np.ndarray) -> tuple[float, float]:
    """Compute Wilcoxon signed-rank test safely.

    Guards against scipy ValueError when all differences are zero (e.g. silent
    sparse features that never fire in any window).
    """
    arr = np.asarray(deltas, dtype=np.float64)
    if len(arr) == 0 or np.all(arr == 0):
        return 0.0, 1.0
    try:
        w_stat, w_p = stats.wilcoxon(arr, alternative="two-sided")
        return float(w_stat), float(w_p)
    except Exception:
        return 0.0, 1.0


def compute_paired_statistics(deltas: np.ndarray) -> StatisticalSummary:
    """Calculate paired differences statistics against null hypothesis H0: Delta = 0.

    Parameters
    ----------
    deltas : np.ndarray
        Array of paired differences across windows or genes (Ablation - Reconstruction).

    Returns
    -------
    StatisticalSummary
        Contains mean, median, t-test, Wilcoxon, and fraction negative.
    """
    arr = np.asarray(deltas, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("Deltas array cannot be empty.")

    mean_val = float(np.mean(arr))
    med_val = float(np.median(arr))
    pct_neg = float(np.mean(arr < 0)) * 100.0

    if len(arr) < 2:
        return StatisticalSummary(
            mean_delta=mean_val,
            median_delta=med_val,
            t_statistic=0.0,
            t_p_value=1.0,
            wilcoxon_w=0.0,
            wilcoxon_p_value=1.0,
            pct_negative=pct_neg,
        )

    if np.all(arr == 0):
        return StatisticalSummary(
            mean_delta=0.0,
            median_delta=0.0,
            t_statistic=0.0,
            t_p_value=1.0,
            wilcoxon_w=0.0,
            wilcoxon_p_value=1.0,
            pct_negative=0.0,
        )

    if len(np.unique(arr)) == 1:
        w_stat, w_p = safe_wilcoxon(arr)
        t_stat = float("inf") if mean_val > 0 else float("-inf")
        t_p = 0.0
        return StatisticalSummary(
            mean_delta=mean_val,
            median_delta=med_val,
            t_statistic=t_stat,
            t_p_value=t_p,
            wilcoxon_w=w_stat,
            wilcoxon_p_value=w_p,
            pct_negative=pct_neg,
        )

    t_stat, t_p = stats.ttest_1samp(arr, 0.0)
    w_stat, w_p = safe_wilcoxon(arr)

    return StatisticalSummary(
        mean_delta=mean_val,
        median_delta=med_val,
        t_statistic=float(t_stat),
        t_p_value=float(t_p),
        wilcoxon_w=w_stat,
        wilcoxon_p_value=w_p,
        pct_negative=pct_neg,
    )
