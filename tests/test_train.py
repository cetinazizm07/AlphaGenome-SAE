"""Training loop: metrics, exact resume, and what ends up in the checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ag_sae import concepts as C
from ag_sae import train as T
from ag_sae.extract import ShardStore, sha256
from ag_sae.sae import Recipe, build


def make_store(directory, dim=8, rows=(96, 96), val_rows=64, seed=0):
    """Write a minimal extraction directory with one tap and two splits."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shards, shard_id = [], 0
    for split, sizes in (("train", rows), ("val", (val_rows,))):
        for size in sizes:
            stem = f"{shard_id:05d}_{split}_tap"
            array = rng.normal(size=(size, dim)).astype(np.float16)
            with open(directory / f"{stem}.npy", "wb") as handle:
                np.save(handle, array, allow_pickle=False)
            starts = np.arange(size) * 128 + shard_id * 100_000
            import pandas as pd
            pd.DataFrame({"chrom": "chr1", "bin_start": starts, "bin_end": starts + 128,
                          "split": split, "window_start": shard_id * 100_000}
                         ).to_parquet(directory / f"{stem}.parquet", index=False)
            shards.append({"shard": shard_id, "split": split, "n_windows": 1, "taps": {"tap": {
                "activations": f"{stem}.npy", "coordinates": f"{stem}.parquet",
                "rows": size, "channels": dim,
                "activations_sha256": sha256(directory / f"{stem}.npy"),
                "coordinates_sha256": sha256(directory / f"{stem}.parquet")}}})
            shard_id += 1
    (directory / "index.json").write_text(json.dumps(
        {"identity": {"version": 1, "seed": seed}, "shards": shards, "complete": True}))
    return ShardStore(directory, "tap", "train"), ShardStore(directory, "tap", "val")


def recipe(**kwargs) -> Recipe:
    base = dict(n_features=32, topk_pct=0.25, lr=1e-2, steps=6, batch_tokens=32,
                microbatch=16, eval_every=3, checkpoint_every=2, seed=1)
    base.update(kwargs)
    return Recipe(**base)


class TestEvaluate:
    def test_fvu_is_one_when_the_model_only_predicts_the_mean(self, tmp_path):
        train_store, val_store = make_store(tmp_path / "cache")
        model = build(train_store.dim, recipe(), np.ones(train_store.dim), "cpu")
        with torch.no_grad():                      # silence the latent path
            model.core.encoder.weight.zero_()
            model.core.latent_bias.zero_()
            model.core.decoder.weight.zero_()
            model.core.pre_bias.zero_()
        metrics = T.evaluate(model, val_store, 16, "cpu")
        # With all latents dead the decoder returns the per-bin mean and std,
        # so the error is the split's own variance: FVU close to 1.
        assert metrics["fvu"] == pytest.approx(1.0, abs=0.25)
        assert metrics["mean_l0"] == 0.0
        assert metrics["features_never_fired"] == model.hidden

    def test_metrics_are_consistent_with_each_other(self, tmp_path):
        train_store, val_store = make_store(tmp_path / "cache")
        model = build(train_store.dim, recipe(), np.ones(train_store.dim), "cpu")
        metrics = T.evaluate(model, val_store, 16, "cpu")
        assert metrics["rows"] == len(val_store)
        assert metrics["features_fired"] + metrics["features_never_fired"] == model.hidden
        assert 0 <= metrics["mean_l0"] <= model.k
        assert metrics["dictionary_used"] == pytest.approx(
            metrics["features_fired"] / model.hidden)


class TestTrainingRun:
    def test_completes_the_step_budget_and_improves_reconstruction(self, tmp_path):
        train_store, val_store = make_store(tmp_path / "cache")
        result = T.train(train_store, val_store, recipe(steps=12, eval_every=4),
                         tmp_path / "run", device="cpu", log=lambda *a: None)
        assert result["status"] == "complete" and result["steps"] == 12
        history = json.loads((tmp_path / "run" / "history.json").read_text())
        assert [h["step"] for h in history] == [4, 8, 12]
        assert history[-1]["validation"]["fvu"] < history[0]["validation"]["fvu"]
        assert (tmp_path / "run" / "best.pt").exists()
        assert (tmp_path / "run" / "training.json").exists()

    def test_best_checkpoint_loads_in_the_analysis_stage(self, tmp_path):
        train_store, val_store = make_store(tmp_path / "cache")
        T.train(train_store, val_store, recipe(), tmp_path / "run",
                device="cpu", log=lambda *a: None)
        frozen = C.FrozenSAE.from_torch_checkpoint(tmp_path / "run" / "best.pt")
        assert frozen.d_in == train_store.dim and frozen.n_features == 32
        codes = frozen.encode(train_store.take(np.arange(16)))
        assert codes.shape == (16, 32)
        assert (np.diff(codes.tocsr().indptr) <= frozen.k).all()

    def test_channel_scale_uses_the_training_split_only(self, tmp_path):
        train_store, val_store = make_store(tmp_path / "cache")
        # A huge value in validation must not change the scale.
        array = np.load(val_store.directory / val_store.records[0]["activations"])
        array[0, 0] = 500
        with open(val_store.directory / val_store.records[0]["activations"], "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        T.train(train_store, ShardStore(val_store.directory, "tap", "val"), recipe(),
                tmp_path / "run", device="cpu", log=lambda *a: None)
        scale = torch.load(tmp_path / "run" / "latest.pt", weights_only=False)["channel_scale"]
        assert scale[0] < 100


class TestResume:
    @pytest.mark.parametrize("stop_at", [2, 5])
    def test_resumed_run_matches_an_uninterrupted_one(self, tmp_path, stop_at):
        train_store, val_store = make_store(tmp_path / "cache")
        settings = recipe(steps=8, eval_every=4)

        whole = T.train(train_store, val_store, settings, tmp_path / "whole",
                        device="cpu", log=lambda *a: None)
        T.train(train_store, val_store, settings, tmp_path / "split",
                device="cpu", stop_after=stop_at, log=lambda *a: None)
        resumed = T.train(train_store, val_store, settings, tmp_path / "split",
                          device="cpu", log=lambda *a: None)

        assert whole["steps"] == resumed["steps"] == 8
        a = torch.load(tmp_path / "whole/latest.pt", weights_only=False)
        b = torch.load(tmp_path / "split/latest.pt", weights_only=False)
        assert a["history"] == b["history"]
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_changing_the_recipe_refuses_to_reuse_the_directory(self, tmp_path):
        train_store, val_store = make_store(tmp_path / "cache")
        T.train(train_store, val_store, recipe(), tmp_path / "run",
                device="cpu", stop_after=2, log=lambda *a: None)
        with pytest.raises(ValueError, match="changed since this run started"):
            T.train(train_store, val_store, recipe(lr=5e-3), tmp_path / "run",
                    device="cpu", log=lambda *a: None)


class TestGuards:
    def test_mismatched_tap_widths_are_rejected(self, tmp_path):
        train_store, _ = make_store(tmp_path / "a", dim=8)
        _, other_val = make_store(tmp_path / "b", dim=6)
        with pytest.raises(ValueError, match="different widths"):
            T.train(train_store, other_val, recipe(), tmp_path / "run", device="cpu")

    def test_gradient_accumulation_must_divide_the_batch(self, tmp_path):
        with pytest.raises(ValueError, match="multiple of microbatch"):
            recipe(batch_tokens=30, microbatch=16).validate(8)
