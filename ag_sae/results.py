"""A campaign's per-window and per-gene outputs as tidy tables.

Every configuration folder (``<fold>_<layer>_<seed>``) of a triad/injection
campaign and of a motif campaign is read into long tables, one row per
measurement, so the analysis never parses JSON again:

* ``triad``: pool, fold, layer, seed, window (chrom, start), concept, kind
  (feature / joint / null), method (reconstruction / error_preserving),
  feature_id, fired_bins (on the edited bins), edit_norm (L2 size of the edit),
  head, delta, baseline, original, relative_change
* ``injection``: pool, fold, layer, seed, gene, dose, modality, status, level,
  code_change, feature_at_tss, paired_log2 (target minus control log2 ratio,
  median over tracks), paired_difference, fraction_target_increase, ...
* ``motif``: pool, fold, layer, seed, window, head, y_wt, y_ism, y_abl,
  y_rescue, delta_ism, delta_abl, delta_rescue, latent values

The first campaign's summary builder looked for keys its JSON never had and
wrote a triad table with no measurements; reading the per-window files
directly avoids that class of error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CONFIG_RE = re.compile(r"^(?P<fold>fold\d+)_(?P<layer>resid_pre_b\d+)_(?P<seed>seed\d+)$")
TRIAD_FILE = "triad_40windows_results.json"
GENE_FILE = "gene_cohort_results.json"
MOTIF_FILE = "motif_per_window_results.json"
#: per-window triad fields -> method name in the table
TRIAD_FIELDS = {"heads": "reconstruction", "heads_error_preserving": "error_preserving"}


def parse_config(name: str) -> dict | None:
    match = CONFIG_RE.match(name)
    return None if match is None else match.groupdict()


def _relative(delta, baseline):
    if delta is None or baseline is None or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return np.nan
    return float(delta) / float(baseline)


def _head_rows(base: dict, heads: dict) -> Iterable[dict]:
    for head, values in heads.items():
        delta = values.get("delta_on_concept")
        baseline = values.get("baseline_on_concept")
        yield {**base, "head": head,
               "delta": np.nan if delta is None else float(delta),
               "baseline": np.nan if baseline is None else float(baseline),
               "original": np.nan if values.get("original_on_concept") is None
               else float(values["original_on_concept"]),
               "relative_change": _relative(delta, baseline)}


def triad_rows(windows: list[dict], config: dict) -> list[dict]:
    rows = []
    for w in windows:
        where = {**config, "window": int(w["window_index"]), "chrom": w["chrom"], "start": int(w["start"])}
        norms = w.get("edit_norms", {})

        def norm(condition):
            return float(norms.get(condition, {}).get("intended", np.nan))

        for concept, entry in w["features"].items():
            for field_name, method in TRIAD_FIELDS.items():
                if field_name in entry:
                    condition = f"ablate_{concept}" + ("_ep" if method == "error_preserving" else "")
                    rows += _head_rows({**where, "concept": concept, "kind": "feature", "method": method,
                                        "feature_id": int(entry["feature_id"]), "null_index": -1,
                                        "fired_bins": entry.get("fired_bins", np.nan),
                                        "edit_norm": norm(condition)}, entry[field_name])
        for concept, entry in w.get("joint", {}).items():
            rows += _head_rows({**where, "concept": concept, "kind": "joint", "method": "error_preserving",
                                "feature_id": -1, "null_index": -1, "fired_bins": np.nan,
                                "edit_norm": norm(f"joint_{concept}_ep")}, entry["heads_error_preserving"])
        for concept, entries in w.get("null", {}).items():
            for j, entry in enumerate(entries):
                rows += _head_rows({**where, "concept": concept, "kind": "null", "method": "error_preserving",
                                    "feature_id": int(entry["feature_id"]), "null_index": j,
                                    "fired_bins": entry.get("fired_bins", np.nan),
                                    "edit_norm": norm(f"null_{concept}_{j}_ep")},
                                   entry["heads_error_preserving"])
    return rows


def gene_rows(genes: list[dict], config: dict) -> list[dict]:
    rows = []
    for g in genes:
        where = {**config, "gene": g["gene_name"], "chrom": g.get("chrom"), "tss0": g.get("tss0"),
                 "feature_at_tss": g.get("feature_at_tss", np.nan)}
        # The first campaign has no "doses": its one dose is the record itself.
        doses = g.get("doses") or {"x1": {"status": g["status"], "effects": g.get("effects"),
                                          "level": np.nan, "code_change": np.nan}}
        for tag, dose in doses.items():
            for modality, eff in (dose.get("effects") or {}).items():
                rows.append({**where, "dose": tag, "multiplier": float(tag[1:]),
                             "status": dose["status"], "level": dose.get("level", np.nan),
                             "code_change": dose.get("code_change", np.nan), "modality": modality,
                             "n_tracks": eff["n_tracks"],
                             "target_log2": eff["target_median_log2_ratio"],
                             "control_log2": eff["control_median_log2_ratio"],
                             "paired_log2": eff["paired_target_minus_control_median"],
                             "paired_difference": eff["paired_difference"],
                             "fraction_target_increase": eff["fraction_target_increase"]})
            if not dose.get("effects"):
                rows.append({**where, "dose": tag, "multiplier": float(tag[1:]), "status": dose["status"],
                             "level": dose.get("level", np.nan), "code_change": dose.get("code_change", np.nan),
                             "modality": None})
    return rows


def motif_rows(windows: list[dict], config: dict) -> list[dict]:
    rows = []
    for i, w in enumerate(windows):
        where = {**config, "window": i, "chrom": w["chrom"], "start": int(w["start"]),
                 "wt_latent": w.get("wt_latent_val", np.nan), "mut_latent": w.get("mut_latent_val", np.nan),
                 "anchor_pvalue": w.get("anchor_pvalue", np.nan)}
        for head, v in w["heads"].items():
            rows.append({**where, "head": head, **{k: v[k] for k in
                         ("y_wt", "y_ism", "y_abl", "y_rescue", "delta_ism", "delta_abl", "delta_rescue")
                         if k in v}})
    return rows


def _configs(root: Path) -> Iterable[tuple[Path, dict]]:
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        config = parse_config(folder.name)
        if config is not None:
            yield folder, config


def collect(campaign_dirs: dict[str, Path] | None = None,
            motif_dirs: dict[str, Path] | None = None) -> dict[str, pd.DataFrame]:
    """Tables from {pool: campaign folder} and {pool: motif folder}; missing files are skipped."""
    triad, genes, motif, discovery = [], [], [], []
    for pool, root in (campaign_dirs or {}).items():
        for folder, config in _configs(root):
            config = {"pool": pool, **config}
            if (folder / TRIAD_FILE).is_file():
                triad += triad_rows(json.loads((folder / TRIAD_FILE).read_text()), config)
            if (folder / GENE_FILE).is_file():
                genes += gene_rows(json.loads((folder / GENE_FILE).read_text()), config)
            if (folder / "feature_discovery.json").is_file():
                record = json.loads((folder / "feature_discovery.json").read_text())
                for concept, feat in record.get("primary", {}).items():
                    discovery.append({**config, "concept": concept, "feature_id": int(feat),
                                      "dose": record.get("injection_dose", {}).get("level", np.nan),
                                      "protocol": record.get("protocol", {}).get("name")})
    for pool, root in (motif_dirs or {}).items():
        for folder, config in _configs(root):
            if (folder / MOTIF_FILE).is_file():
                motif += motif_rows(json.loads((folder / MOTIF_FILE).read_text()), {"pool": pool, **config})
    return {"triad": pd.DataFrame(triad), "injection": pd.DataFrame(genes),
            "motif": pd.DataFrame(motif), "discovery": pd.DataFrame(discovery)}


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        if frame.empty:
            continue
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
    return written


def read_tables(out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    return {p.stem: pd.read_parquet(p) for p in sorted(out_dir.glob("*.parquet"))}
