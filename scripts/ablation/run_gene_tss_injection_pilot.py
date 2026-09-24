#!/usr/bin/env python3
"""One full-stream, one-TSS-bin gene-expression injection pilot.

The gene, 128-bp TSS bin, transcript exons, and assay tracks come from a
hash-frozen plan selected without model outcomes. Only that TSS bin is edited.
The measured result is AlphaGenome's predicted gene expression, not a wet-lab
expression change or a claim about every gene.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from fullstream_local_edit import local_edit_triplet
from gene_tss_readout import (
    extract_gene_readouts,
    match_control_norm,
    summarize_gene_effects,
)
from plan_fullstream_cohort import sha256
from plan_gene_tss_injection import select_tracks
from run_cloud_ablation_pilot import chrom_sequence, load_sampled_coordinates
from run_cloud_fullstream_control_pilot import FullStreamTower, run_replacement


HEADS = ("rna_seq", "cage", "procap")


def checked_inputs(
    plan_path: Path,
    acts_dir: Path,
    fasta_dir: Path,
    parent_override: Path | None = None,
    gtf_override: Path | None = None,
    source_overrides: dict[str, Path] | None = None,
):
    """Validate frozen source bytes, held-out coordinates, gene, and DNA span."""
    plan = json.loads(plan_path.read_text())
    if (plan.get("format") != "fullstream_gene_tss_injection_plan_v1"
            or plan.get("fold") != "fold1" or plan.get("sae_seed") != 0
            or plan.get("tap") != "resid_pre_b8"
            or plan.get("target_feature") != 199
            or plan.get("control_feature") != 4247):
        raise ValueError("Unexpected frozen gene TSS protocol")
    parent = parent_override or Path(plan["parent_plan_path"])
    gtf = gtf_override or Path(plan["gtf_path"])
    if sha256(parent) != plan["parent_plan_sha256"] or sha256(gtf) != plan["gtf_sha256"]:
        raise ValueError("Parent plan or GENCODE annotation changed")
    paths = {name: Path(value) for name, value in plan["source_paths"].items()}
    for name, path in (source_overrides or {}).items():
        if name not in paths:
            raise ValueError(f"Unknown frozen source: {name}")
        paths[name] = path
    for name, path in paths.items():
        if sha256(path) != plan["source_sha256"][name]:
            raise ValueError(f"Frozen source changed: {name}")
    gene = plan["gene"]
    chrom, start, end = gene["chrom"], int(gene["window_start"]), int(gene["window_end"])
    if (end - start != 1_048_576
            or not start <= gene["tss0"] < end
            or (gene["tss0"] - start) // 128 != gene["tss_bin_index"]
            or gene["strand"] not in {"+", "-"}):
        raise ValueError("Gene TSS does not map to the frozen 128-bp window")
    if select_tracks(gene["strand"]) != plan["tracks"]:
        raise ValueError("Installed model track metadata differs from frozen plan")

    manifest = pd.read_parquet(paths["manifest"])
    held = manifest.loc[(manifest.split == "test") & (manifest.chrom == chrom)
                        & (manifest.win_start == start)]
    if len(held) != 1 or int(held.iloc[0].win_end) != end:
        raise ValueError("Gene window is not uniquely held out")
    ann = pd.read_parquet(paths["annotations"])
    bins = ann.loc[(ann.split == "test") & (ann.chrom == chrom)
                   & (ann.bin_start >= start) & (ann.bin_start < end)]
    bins = bins.sort_values("bin_start").reset_index(drop=True)
    expected = start + 128 * np.arange(8192, dtype=np.int64)
    if (not np.array_equal(bins.bin_start.to_numpy(dtype=np.int64), expected)
            or not bins.n_mask.to_numpy(dtype=bool).all()
            or bool(bins.iloc[gene["tss_bin_index"]].cCRE_PLS)):
        raise ValueError("TSS bin is not on the frozen complete, PLS-negative test grid")
    coords, coord_path = load_sampled_coordinates(acts_dir, "resid_pre_b8", chrom, start)
    if (len(coords) != 8192 or not (coords.split == "test").all()
            or not np.array_equal(coords.bin_start.to_numpy(dtype=np.int64), expected)):
        raise ValueError("Native sampled coordinates differ from 128-bp annotation grid")
    index = json.loads(paths["activation_index"].read_text())
    relative = str(coord_path.relative_to(acts_dir))
    indexed = [shard["taps"]["resid_pre_b8"] for shard in index["shards"]
               if shard["split"] == "test" and "resid_pre_b8" in shard["taps"]
               and shard["taps"]["resid_pre_b8"]["coordinates"] == relative]
    coord_sha = sha256(coord_path)
    if len(indexed) != 1 or indexed[0]["coordinates_sha256"] != coord_sha:
        raise ValueError("Native coordinate hash differs from activation index")

    from alphagenome_pytorch.utils import sequence_to_onehot
    sequence = chrom_sequence(fasta_dir, chrom)[start:end]
    if len(sequence) != 1_048_576:
        raise ValueError("FASTA sequence does not cover the entire gene window")
    onehot = sequence_to_onehot(sequence).astype(np.float32)
    if onehot.shape != (1_048_576, 4) or not onehot.sum(-1).reshape(8192, 128).all():
        raise ValueError("Ambiguous or malformed DNA in the selected window")
    return plan, paths, bins, onehot, coord_sha


def parse_overrides(items: list[str]) -> dict[str, Path]:
    """Support moving identical checkpoints and annotations to another VM."""
    overrides = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in overrides:
            raise ValueError(f"Expected unique NAME=PATH override, got {item!r}")
        overrides[name] = Path(value)
    return overrides


def write_result(out: Path, plan_path: Path, receipt: dict,
                 readouts: dict[str, np.ndarray] | None) -> None:
    """Write one immutable receipt and, when measurable, compact track values."""
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, out / "plan.json")
    for source in (__file__, Path(__file__).with_name("gene_tss_readout.py"),
                   Path(__file__).with_name("fullstream_local_edit.py")):
        shutil.copy2(source, out / Path(source).name)
    if readouts is not None:
        temporary = out / "readouts.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **readouts)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(out / "readouts.npz")
        receipt["readouts_sha256"] = sha256(out / "readouts.npz")
    else:
        receipt["readouts_sha256"] = None
    temporary = out / "receipt.json.tmp"
    with temporary.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out / "receipt.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path)
    parser.add_argument("--gtf", type=Path)
    parser.add_argument("--source-override", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.out.exists() or args.batch_size < 1:
        raise ValueError("Output already exists or batch size is invalid")
    plan, paths, bins, onehot, coord_sha = checked_inputs(
        args.plan, args.acts_dir, args.fasta_dir, args.parent_plan,
        args.gtf, parse_overrides(args.source_override),
    )

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    tap = TAPS["resid_pre_b8"]
    if tap.kind != "tower" or tap.bin_bp != 128:
        raise ValueError("Expected 128-bp tower-block input")
    sae = BorzoiSAE.from_checkpoint(paths["sae"], device="cuda")
    if sae.d_in != tap.channels or sae.hidden <= 4247:
        raise ValueError("SAE dimensions differ from frozen plan")
    model = AlphaGenome.from_pretrained(
        paths["weights"], dtype_policy=DtypePolicy.mixed_precision(), device="cuda"
    )
    model.eval().requires_grad_(False)
    input_tensor = torch.from_numpy(onehot)[None].to("cuda")
    organism = torch.tensor([0], device="cuda")
    forward_args = {"resolutions": (128,), "heads": HEADS, "channels_last": True}
    gene = plan["gene"]
    tss_bin = int(gene["tss_bin_index"])
    mask = np.zeros(8192, dtype=bool)
    mask[tss_bin] = True
    captured = []
    readouts: dict[str, np.ndarray] = {}
    started = time.monotonic()

    def capture(value):
        captured.append(np.array(value, dtype=np.float32, copy=True))
        return value

    def record(output, label):
        selected = extract_gene_readouts(output, gene, plan["tracks"])
        readouts.update({f"{head}_{label}": values for head, values in selected.items()})

    with torch.inference_mode(), FullStreamTower(model, tap, capture):
        original = model.predict(input_tensor, organism, **forward_args)
    if len(captured) != 1 or captured[0].shape != (8192, tap.channels):
        raise ValueError("Unexpected full-stream tower capture")
    record(original, "original")
    del original
    activation = torch.from_numpy(captured[0])[None].to("cuda")
    del captured

    baseline, target, control, counts = local_edit_triplet(
        sae, activation, mask, 199, 4247, "inject",
        plan["target_dev_positive_p95"], plan["control_dev_positive_p95"],
        args.batch_size,
    )
    receipt = {
        "format": "fullstream_gene_tss_injection_pilot_v1",
        "interpretation": plan["interpretation"],
        "plan_sha256": sha256(args.plan),
        "worker_sha256": sha256(Path(__file__)),
        "readout_code_sha256": sha256(Path(__file__).with_name("gene_tss_readout.py")),
        "local_edit_code_sha256": sha256(Path(__file__).with_name("fullstream_local_edit.py")),
        "gene": gene, "fold": "fold1", "sae_seed": 0, "tap": "resid_pre_b8",
        "target_feature": 199, "control_feature": 4247,
        "tss_bin_start": int(bins.iloc[tss_bin].bin_start),
        "n_intervened_bins": 1,
        "intervention_scope": "single MANE Select TSS 128-bp bin",
        "control_rule": "same TSS bin; control decoder direction L2-matched before output inspection",
        "code_counts": counts,
        "source_sha256": plan["source_sha256"],
        "gtf_sha256": plan["gtf_sha256"],
        "sampled_coordinates_sha256": coord_sha,
        "tracks": plan["tracks"],
    }
    try:
        control, norms = match_control_norm(baseline, target, control, tss_bin)
    except ValueError as error:
        receipt.update({"status": "unmeasurable", "reason": str(error),
                        "elapsed_seconds": time.monotonic() - started})
        write_result(args.out, args.plan, receipt, None)
        print(json.dumps({"out": str(args.out), "status": receipt["status"],
                          "reason": receipt["reason"],
                          "receipt_sha256": sha256(args.out / "receipt.json")}))
        return
    receipt["tap_intervention_l2"] = norms
    for label, replacement in (("reconstruction", baseline),
                               ("target", target), ("control", control)):
        with torch.inference_mode(), FullStreamTower(model, tap, run_replacement(replacement)):
            output = model.predict(input_tensor, organism, **forward_args)
        record(output, label)
        del output
        torch.cuda.empty_cache()
    receipt["status"] = "measured"
    receipt["effects"] = summarize_gene_effects(readouts, plan["tracks"])
    receipt["elapsed_seconds"] = time.monotonic() - started
    write_result(args.out, args.plan, receipt, readouts)
    print(json.dumps({"out": str(args.out), "status": receipt["status"],
                      "gene": gene["gene_name"], "effects": receipt["effects"],
                      "receipt_sha256": sha256(args.out / "receipt.json"),
                      "readouts_sha256": receipt["readouts_sha256"]}))


if __name__ == "__main__":
    main()
