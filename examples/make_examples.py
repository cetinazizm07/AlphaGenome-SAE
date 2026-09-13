"""Render every figure in the module from synthetic data.

Nothing here touches a real run. The numbers are invented so the shapes are
obvious: recovery is clear for promoters and absent for the shuffled control,
depth rises for the contextual concept and is flat for the local one, and one
feature carries a planted motif. Use it to check layout and to see what each
figure is meant to say, never as a result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from ag_sae import figures as fig

TAPS = list(fig.TAP_ORDER)
CONCEPTS = ["cCRE_PLS", "cCRE_pELS", "cCRE_dELS", "cCRE_CTCF-only",
            "splice_donor", "TSS_within_2kb", "GC_rich", "shuffled_dELS"]
#: How each concept is meant to behave with depth, so the demo shows the
#: contrast the real study is looking for.
SHAPE = {"cCRE_PLS": "rising", "cCRE_pELS": "rising", "cCRE_dELS": "rising",
         "cCRE_CTCF-only": "flat", "splice_donor": "flat",
         "TSS_within_2kb": "rising", "GC_rich": "flat", "shuffled_dELS": "null"}


def depth_curve(shape: str, rng: np.random.Generator) -> np.ndarray:
    """AUROC across the six taps, shallow to deep."""
    steps = np.linspace(0, 1, len(TAPS))
    if shape == "rising":
        base = 0.58 + 0.30 * steps ** 1.6
    elif shape == "flat":
        base = 0.80 - 0.04 * steps
    else:
        base = 0.50 + 0.01 * steps
    return np.clip(base + rng.normal(0, 0.012, len(TAPS)), 0.4, 0.99)


def tidy_depth(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for concept in CONCEPTS:
        auroc = depth_curve(SHAPE[concept], rng)
        for tap, value in zip(TAPS, auroc):
            null = 0.55 if SHAPE[concept] != "null" else 0.56
            rows.append({
                "tap": tap, "concept": concept, "best_feature": int(rng.integers(0, 8192)),
                "best_auroc": float(value),
                "raw_best_auroc": float(np.clip(value - rng.uniform(0.02, 0.10), 0.4, 1)),
                "null_p95": null, "recovered": bool(value > null),
            })
    return pd.DataFrame(rows)


def synthetic_feature(rng: np.random.Generator, n: int = 3000, length: int = 16):
    """A feature that fires on a planted motif, plus a contaminated tail.

    The strongest sites all carry the motif; the weak ones mostly do not. That
    is exactly the situation the spread panel of the feature card exists to
    expose.
    """
    letters = np.array(list("ACGT"))
    sequences = ["".join(rng.choice(letters, length)) for _ in range(n)]
    activation = np.zeros(n)
    strong = rng.choice(n, 90, replace=False)
    for index in strong:
        start = length // 2 - 3
        sequences[index] = sequences[index][:start] + "TGASTCA"[:6] + sequences[index][start + 6:]
    activation[strong] = rng.gamma(6.0, 0.9, strong.size)
    weak = rng.choice(np.setdiff1d(np.arange(n), strong), 160, replace=False)
    activation[weak] = rng.gamma(1.2, 0.5, weak.size)
    return activation, sequences


INDEX = """# Figure examples, synthetic data

Every number here is invented. The point is the shape of each figure and
whether the layout holds, never the values.

| File | Question it answers |
|---|---|
| 01_recovery | Which concepts does the dictionary recover, and by how much does it beat the raw channels? |
| 02_depth | Where in the network is each concept built? Flat means convolution already had it, rising means attention added it. |
| 03_locus | What does one feature actually do along the genome, against the annotation? |
| 04_specificity | Are features concentrated on one concept, or spread across many? |
| 05_quality_fixed_sparsity | How well does each tap reconstruct, and how much does the seed matter? |
| 06_pareto_sparsity_sweep | The same figure if sparsity were swept instead of pinned by TopK. Not our design; included to show the other layout. |
| 07_feature_card | Everything about one feature. The second logo is the honest one: if it disagrees with the first, the feature is not monosemantic. |
| 08_seed_stability | How much of the dictionary survives a change of seed, against what random directions would give. |

Each figure ships as PDF, SVG, 300 dpi PNG and a CSV of its numbers. The SVG
keeps text as text, so it opens editable in Illustrator.

Rebuild with `PYTHONPATH=. python examples/make_examples.py`.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path.home() / "Desktop" / "ag_sae_figure_examples"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    written = []

    # 1. recovery ---------------------------------------------------------
    tidy = tidy_depth(rng)
    deepest = tidy[tidy.tap == TAPS[-1]].copy()
    per_concept = deepest[["concept", "best_feature", "best_auroc",
                           "raw_best_auroc", "null_p95", "recovered"]].copy()
    per_concept["n_positive_bins"] = rng.integers(200, 9000, len(per_concept))
    per_concept["sae_minus_raw"] = per_concept.best_auroc - per_concept.raw_best_auroc
    written += fig.figure_recovery(per_concept, out / "01_recovery",
                                   title="Concept recovery at the deepest tap")

    # 2. depth ------------------------------------------------------------
    by_tap = {tap: frame.drop(columns="tap") for tap, frame in tidy.groupby("tap")}
    written += fig.figure_depth(by_tap, out / "02_depth")

    # 3. locus ------------------------------------------------------------
    n_bins = 4096
    starts = np.arange(n_bins) * 128
    bins = pd.DataFrame({"chrom": "chr1", "bin_start": starts, "bin_end": starts + 128})
    profile = np.clip(rng.gamma(0.4, 0.5, n_bins), 0, None)
    # cCRE intervals in the same shape read_ccre_bed returns: class -> chrom ->
    # (n, 2) start/end pairs in base pairs.
    ccre = {}
    for name in ("PLS", "pELS", "dELS"):
        centres = np.sort(rng.choice(n_bins, 40, replace=False))
        widths = rng.integers(1, 4, centres.size)
        spans = np.stack([centres * 128, (centres + widths) * 128], axis=1)
        ccre[name] = {"chr1": spans.astype(np.int64)}
        for centre, width in zip(centres, widths):
            profile[centre:centre + width] += rng.gamma(3.0, 0.8, width)
    written += fig.figure_locus(bins, profile, ccre, out / "03_locus", feature=1423)

    # 4. specificity ------------------------------------------------------
    n_features = 8192
    per_feature = pd.DataFrame({
        "feature": np.arange(n_features),
        "dominant_concept": rng.choice(CONCEPTS, n_features),
        "dominant_share": np.clip(rng.beta(2.2, 3.0, n_features), 0, 1),
    })
    written += fig.figure_specificity(per_feature, out / "04_specificity", chance=0.28)

    # 5. reconstruction quality ------------------------------------------
    fixed = pd.DataFrame([
        {"tap": tap, "seed": seed, "mean_l0": 409.0,
         "fvu": 0.30 - 0.02 * i + rng.normal(0, 0.008)}
        for i, tap in enumerate(TAPS) for seed in range(3)])
    written += fig.figure_pareto(fixed, out / "05_quality_fixed_sparsity",
                                 title="Reconstruction by depth, three seeds each")

    sweep = pd.DataFrame([
        {"tap": tap, "seed": seed, "mean_l0": float(k),
         "fvu": float(0.9 * np.exp(-k / 500) + 0.06 + rng.normal(0, 0.01))}
        for tap in TAPS[3:] for seed in range(2)
        for k in (40, 80, 160, 320, 640, 1280)])
    written += fig.figure_pareto(sweep, out / "06_pareto_sparsity_sweep",
                                 title="If sparsity is swept instead of fixed")

    # 6. feature card -----------------------------------------------------
    activation, sequences = synthetic_feature(rng)
    written += fig.figure_feature_card(
        1423, activation, sequences, out / "07_feature_card",
        concept_auroc={"cCRE_PLS": 0.88, "cCRE_pELS": 0.71, "cCRE_dELS": 0.58,
                       "splice_donor": 0.52, "shuffled_dELS": 0.50})

    # 7. seed stability ---------------------------------------------------
    shared = rng.normal(size=(2048, 64))
    decoders = {}
    for seed in (0, 1, 2):
        noise = rng.normal(size=shared.shape)
        # Two thirds of the dictionary is reproducible, the rest is not.
        keep = rng.random(len(shared)) < 0.66
        decoders[seed] = np.where(keep[:, None], shared + 0.35 * noise, noise)
    written += fig.figure_seed_stability(decoders, out / "08_seed_stability")

    (out / "README.md").write_text(INDEX)
    print(f"{len(written)} files in {out}")
    for path in sorted(out.iterdir()):
        print("  ", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
