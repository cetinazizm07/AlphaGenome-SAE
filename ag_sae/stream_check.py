"""Is the stream the SAEs were trained on the stream the experiments edit?

Extraction (ag_sae.extract) captured the residual stream with a pre-hook on a
block's `mha` during ``model.encode(...)`` and stored it as float16. The
interventions read and edit the same point inside ``model.predict(...)``,
which wraps the forward in bf16 autocast. If the two streams differ, every
experiment feeds the SAE input it was not trained on and the edited features
are not quite the features that were trained.

This runs a few windows through both paths and reports, per window and
pooled:

* ``onehot_max_abs_diff``: extraction's one-hot (extract.one_hot, N -> 0)
  against the experiments' (intervene.sequence_to_onehot)
* ``encode_dtype`` / ``predict_dtype``: dtype of the stream in each path
* ``mha_vs_block_entry``: the predict-path stream seen by the extraction hook
  (mha pre-hook) against the one the edits replace (entrance of
  _forward_block); anything the block applies before mha shows up here

* ``rel_l2``: ||predict - encode|| / ||encode|| over the window's rows
* ``row_cos_min`` / ``row_cos_mean``: cosine between matching rows
* ``fvu_encode`` / ``fvu_predict``: SAE FVU on each stream (against the
  window's own channel means, so compare the two, not with the training FVU)
* ``support_jaccard``: mean overlap of the TopK supports row by row
* ``code_rel_l1``: ||codes_predict - codes_encode||_1 / ||codes_encode||_1

Usage on the VM::

    python -m ag_sae.stream_check --fold fold3 --layer resid_pre_b8 \\
        --sae-pool sae_v2 --seed seed0 --n-windows 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ag_sae.edit import ResidualEdit, capture_residual
from ag_sae.extract import Capture, one_hot


def _compute_autocast(model: Any, device: torch.device):
    """The autocast predict() opens: on unless the model's policy computes in float32.

    alphagenome_pytorch 0.3.1's encode() casts its input to the compute dtype
    but opens no autocast, so under the mixed-precision policy it fails on its
    first convolution (bf16 input, fp32 weights). The extraction that made the
    shards ran in bf16 with an earlier version; this is the closest call that
    runs with 0.3.1, and the shard comparison says how close it is.
    """
    policy = getattr(model, "dtype_policy", None)
    dtype = getattr(policy, "compute_dtype", torch.float32)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)


@torch.inference_mode()
def encode_stream(model: Any, tap: Any, x: torch.Tensor) -> tuple[torch.Tensor, str]:
    """The tap as extraction saw it: model.encode with the extraction hooks, float16."""
    organism = torch.zeros(len(x), dtype=torch.long, device=x.device)
    with Capture(model, [tap]) as capture, _compute_autocast(model, x.device):
        model.encode(x, organism_index=organism, resolutions=(128,))
        stream = capture.outputs(0)[tap.name]
    # The shards stored float16 rows (extract._keep_rows).
    return stream.float().half().float(), str(stream.dtype).removeprefix("torch.")


@torch.inference_mode()
def predict_stream(model: Any, tap: Any, x: torch.Tensor) -> tuple[torch.Tensor, str, float]:
    """The tap as the experiments edit it: the stream entering the block inside predict().

    Recorded where ResidualEdit replaces it (as run_conditions does). Also
    returns how far the mha pre-hook's view (what capture_residual and the
    extraction hooks read) is from it.
    """
    seen: dict[str, torch.Tensor] = {}

    def record(stream: torch.Tensor) -> torch.Tensor:
        seen["entry"] = stream.clone()
        return stream

    with ResidualEdit(model, int(tap.key), record):
        model.predict(x, 0, resolutions=(128,), embeddings_only=True)
    entry = seen["entry"][0]
    hooked = capture_residual(model, int(tap.key), x)[0]
    gap = float((hooked.float() - entry.float()).abs().max())
    return entry.float(), str(entry.dtype).removeprefix("torch."), gap


@torch.no_grad()
def compare_streams(sae: Any, reference: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Agreement of two (rows, D) streams and of the SAE's view of them."""
    if reference.shape != other.shape or reference.ndim != 2:
        raise ValueError(f"streams must be matching (rows, D), got {tuple(reference.shape)} "
                         f"and {tuple(other.shape)}")
    a, b = reference.float(), other.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)

    def fvu(raw: torch.Tensor) -> float:
        recon, _codes, scaled = sae(raw)
        variance = (scaled - scaled.mean(0)).square().sum()
        return float((recon - scaled).square().sum() / variance)

    codes_a, _ = sae.encode_codes(a)
    codes_b, _ = sae.encode_codes(b)
    on_a, on_b = codes_a > 0, codes_b > 0
    union = (on_a | on_b).sum(1).clamp_min(1)
    jaccard = (on_a & on_b).sum(1) / union
    return {
        "rows": int(len(a)),
        "rel_l2": float((b - a).norm() / a.norm().clamp_min(1e-12)),
        "row_cos_min": float(cos.min()),
        "row_cos_mean": float(cos.mean()),
        "fvu_encode": fvu(a),
        "fvu_predict": fvu(b),
        "support_jaccard": float(jaccard.mean()),
        "code_rel_l1": float((codes_b - codes_a).abs().sum() / codes_a.abs().sum().clamp_min(1e-12)),
    }


def pooled(results: Sequence[dict[str, float]]) -> dict[str, float]:
    """Row-weighted means across windows; minima stay minima."""
    rows = np.array([r["rows"] for r in results], dtype=np.float64)
    out: dict[str, float] = {"rows": int(rows.sum()), "windows": len(results)}
    for key in results[0]:
        if key == "rows":
            continue
        values = np.array([r[key] for r in results], dtype=np.float64)
        out[key] = float(values.min()) if key.endswith("_min") else float((values * rows).sum() / rows.sum())
    return out


def window_groups(coordinates: Any) -> list[tuple[str, int, np.ndarray]]:
    """(chrom, window_start, row indices) per window of a shard's coordinate table, in order."""
    groups = []
    for (chrom, start), rows in coordinates.groupby(["chrom", "window_start"], sort=False):
        groups.append((str(chrom), int(start), rows.index.to_numpy()))
    return groups


def run_shard_check(model: Any, sae: Any, tap: Any, coordinates: Any, rows: np.ndarray,
                    fetch: Any, n_windows: int, window_bp: int,
                    device: str | torch.device = "cpu") -> dict:
    """Compare the stored training rows of a shard with what each path computes now.

    `coordinates` is the shard's coordinate table (chrom, bin_start,
    window_start, ...; one row per stored row) and `rows` its (n, D) float16
    activations. `fetch(chrom, start, end)` returns the DNA. For each of the
    first `n_windows` windows both paths are run and the stored rows are
    compared with the rows at the same bins.
    """
    from ag_sae.intervene import sequence_to_onehot

    coordinates = coordinates.reset_index(drop=True)
    paths: dict[str, list] = {"predict": [], "encode": []}
    for chrom, start, index in window_groups(coordinates)[:n_windows]:
        sequence = fetch(chrom, start, start + window_bp)
        bins = ((coordinates.bin_start.to_numpy()[index] - start) // tap.bin_bp).astype(np.int64)
        stored = torch.from_numpy(np.asarray(rows[index], dtype=np.float32)).to(device)
        x_edit = torch.from_numpy(np.asarray(sequence_to_onehot(sequence), dtype=np.float32))[None].to(device)
        x_extract = torch.from_numpy(one_hot(sequence))[None].to(device)
        predicted, _dtype, _gap = predict_stream(model, tap, x_edit)
        encoded, _dtype = encode_stream(model, tap, x_extract)
        where = torch.as_tensor(bins, device=device)
        paths["predict"].append(compare_streams(sae, stored, predicted[where].half().float()))
        paths["encode"].append(compare_streams(sae, stored, encoded[where]))
    return {name: {"windows": results, "pooled": pooled(results)} for name, results in paths.items()}


def run_stream_check(model: Any, sae: Any, tap: Any, sequences: Sequence[str],
                     device: str | torch.device = "cpu") -> dict:
    """Compare both paths on each DNA sequence, each with its own one-hot encoding."""
    from ag_sae.intervene import sequence_to_onehot

    windows, dtypes = [], set()
    for sequence in sequences:
        x_extract = torch.from_numpy(one_hot(sequence))[None].to(device)
        x_edit = torch.from_numpy(np.asarray(sequence_to_onehot(sequence), dtype=np.float32))[None].to(device)
        a, encode_dtype = encode_stream(model, tap, x_extract)
        b, predict_dtype, hook_gap = predict_stream(model, tap, x_edit)
        result = compare_streams(sae, a, b)
        result["onehot_max_abs_diff"] = float((x_extract - x_edit).abs().max())
        result["mha_vs_block_entry"] = hook_gap
        windows.append(result)
        dtypes.add((encode_dtype, predict_dtype))
    summary = pooled(windows)
    for key in ("onehot_max_abs_diff", "mha_vs_block_entry"):
        summary[key] = max(w[key] for w in windows)
    summary["encode_dtype"] = sorted({d[0] for d in dtypes})
    summary["predict_dtype"] = sorted({d[1] for d in dtypes})
    return {"windows": windows, "pooled": summary}


def main(argv: Sequence[str] | None = None) -> int:
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    import pandas as pd

    from ag_sae.extract import TAPS
    from ag_sae.intervene import window_sequence
    from ag_sae.sae import BorzoiSAE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", default="fold3")
    parser.add_argument("--layer", default="resid_pre_b8", choices=list(TAPS))
    parser.add_argument("--seed", default="seed0")
    parser.add_argument("--sae-pool", default="sae_v2")
    parser.add_argument("--sae", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fasta", type=Path, default=Path("/mnt/ag/data/GRCh38.primary_assembly.genome.fa"))
    parser.add_argument("--n-windows", type=int, default=4)
    parser.add_argument("--float32", action="store_true",
                        help="load AlphaGenome in full float32 (default: mixed precision, as extraction did)")
    parser.add_argument("--shard-dir", type=Path, default=None,
                        help="activation shards (index.json); also compare against the stored rows")
    parser.add_argument("--shard", default=None,
                        help="shard file stem such as 00029_test; default the first test shard")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    fold_num = args.fold.replace("fold", "")
    weights = args.weights or Path(f"/mnt/ag/weights/model_fold_{fold_num}.safetensors")
    sae_path = args.sae or Path(f"/mnt/ag/{args.sae_pool}/{args.fold}/{args.layer}_{args.seed}/best.pt")
    manifest_path = args.manifest or next(p for p in [
        Path(f"/mnt/ag/analysis-code-a1f31ce/manifests/window_manifest_fold{fold_num}.parquet"),
        Path(f"/mnt/ag/manifests/window_manifest_fold{fold_num}.parquet")] if p.is_file())
    out = args.out or Path(f"/mnt/ag/analysis_runs/stream_check/{args.sae_pool}_{args.fold}_{args.layer}_{args.seed}.json")

    policy = DtypePolicy.full_float32() if args.float32 else DtypePolicy.mixed_precision()
    model = AlphaGenome.from_pretrained(str(weights), dtype_policy=policy, device="cuda")
    model.eval().requires_grad_(False)
    sae = BorzoiSAE.from_checkpoint(sae_path, device="cuda")
    tap = TAPS[args.layer]

    manifest = pd.read_parquet(manifest_path)
    rows = manifest.loc[manifest.split == "test"].head(args.n_windows)
    sequences = [window_sequence(args.fasta, r.chrom, int(r.win_start), int(r.win_end))
                 for r in rows.itertuples()]
    report = run_stream_check(model, sae, tap, sequences, device="cuda")
    if args.shard_dir is not None:
        stem = args.shard or sorted(p.name.removesuffix(f"_{args.layer}.npy")
                                    for p in args.shard_dir.glob(f"*_test_{args.layer}.npy"))[0]
        coordinates = pd.read_parquet(args.shard_dir / f"{stem}_{args.layer}.parquet")
        stored = np.load(args.shard_dir / f"{stem}_{args.layer}.npy", mmap_mode="r")
        window_bp = int((manifest.win_end - manifest.win_start).iloc[0])
        shard = run_shard_check(model, sae, tap, coordinates, stored,
                                lambda c, s, e: window_sequence(args.fasta, c, s, e),
                                args.n_windows, window_bp, device="cuda")
        report["vs_shard"] = {"shard": stem, **shard}
        print("vs stored shard rows:", json.dumps({k: v["pooled"] for k, v in shard.items()}, indent=2))
    report.update({"sae": str(sae_path), "weights": str(weights), "layer": args.layer,
                   "policy": "float32" if args.float32 else "mixed_precision",
                   "windows_used": [f"{r.chrom}:{r.win_start}-{r.win_end}" for r in rows.itertuples()]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pooled"], indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
