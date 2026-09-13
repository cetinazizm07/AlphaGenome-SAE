"""SAE wrapper: recipe arithmetic, channel scaling, and the checkpoint contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ag_sae import concepts as C
from ag_sae import sae as S
from ag_sae.vendor.sae_borzoi import SparseAutoencoder


class TestRecipe:
    def test_two_readings_of_five_percent(self):
        recipe = S.Recipe(expansion=4, topk_pct=0.05)
        assert recipe.hidden(3072) == 12288
        assert recipe.k(3072) == 614                      # 5% of the dictionary
        from dataclasses import replace
        assert replace(recipe, topk_basis="input").k(3072) == 153   # 5% of inputs

    def test_fixed_width_overrides_expansion(self):
        recipe = S.Recipe(expansion=4, n_features=8192)
        assert recipe.hidden(1024) == recipe.hidden(1536) == 8192
        assert recipe.k(1024) == recipe.k(1536) == 409

    @pytest.mark.parametrize("bad", [
        {"topk_pct": 0}, {"topk_pct": 1.5}, {"lr": 0}, {"steps": 0},
        {"topk_basis": "other"}, {"microbatch": 99999},
        {"batch_tokens": 1000, "microbatch": 300}, {"n_features": -1},
    ])
    def test_invalid_settings_are_rejected(self, bad):
        with pytest.raises(ValueError):
            S.Recipe(**bad).validate(1536)

    def test_resolved_records_what_was_actually_used(self):
        resolved = S.Recipe(n_features=8192).resolved(1536)
        assert resolved["hidden"] == 8192 and resolved["d_in"] == 1536
        assert resolved["k"] == 409 and resolved["token_layernorm"] is True


class TestChannelScale:
    def test_uses_the_maximum_per_channel(self):
        batch = np.array([[1.0, 5.0], [3.0, 2.0]])
        scale, fallback = S.channel_scale([batch], 2)
        np.testing.assert_allclose(scale, [3.0, 5.0])
        assert fallback == 0

    def test_signed_and_zero_channels_fall_back(self):
        # Channel 0 is all negative, channel 1 all zero: neither has a usable max.
        batch = np.array([[-4.0, 0.0], [-2.0, 0.0]])
        scale, fallback = S.channel_scale([batch], 2)
        np.testing.assert_allclose(scale, [4.0, 1.0])
        assert fallback == 2

    def test_accumulates_across_batches(self):
        scale, _ = S.channel_scale([np.array([[1.0]]), np.array([[7.0]])], 1)
        assert scale[0] == 7.0

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError, match="non-finite"):
            S.channel_scale([np.array([[np.inf]])], 1)


class TestModel:
    def test_forward_matches_the_unmodified_upstream_class(self):
        """The wrapper must not change the maths, only the input scaling."""
        torch.manual_seed(4)
        scale = torch.linspace(1, 3, 12)
        ours = S.BorzoiSAE(12, 48, 3, scale)
        upstream = SparseAutoencoder(12, 48, 3, sparsity_method="topk_o", normalize=True)
        upstream.load_state_dict(ours.core.state_dict())

        raw = torch.randn(9, 12)
        expected, codes, _ = upstream(raw / scale)
        recon, ours_codes, scaled = ours(raw)
        torch.testing.assert_close(recon, expected, rtol=0, atol=0)
        torch.testing.assert_close(ours_codes, codes, rtol=0, atol=0)
        torch.testing.assert_close(scaled, raw / scale)

    def test_sparsity_and_nonnegativity(self):
        model = S.build(16, S.Recipe(n_features=64, seed=1), np.ones(16))
        _, codes, _ = model(torch.randn(20, 16))
        assert (codes >= 0).all()
        assert ((codes > 0).sum(-1) <= model.k).all()

    def test_reconstruct_returns_raw_units(self):
        scale = np.linspace(1, 4, 8).astype(np.float32)
        model = S.BorzoiSAE(8, 32, 2, scale)
        raw = torch.randn(5, 8)
        back, _ = model.reconstruct(raw)
        recon, _, _ = model(raw)
        torch.testing.assert_close(back, recon * torch.as_tensor(scale))

    @pytest.mark.parametrize("scale", [np.zeros(4), np.array([1.0, -1, 1, 1]), np.ones(3)])
    def test_bad_channel_scale_is_rejected(self, scale):
        with pytest.raises(ValueError, match="channel scale|dimensions"):
            S.BorzoiSAE(4, 16, 2, scale)


class TestCheckpointContract:
    def test_saved_model_reloads_and_encodes_identically(self, tmp_path):
        """The training output must be readable by the analysis stage."""
        rng = np.random.default_rng(0)
        activations = rng.normal(size=(64, 12)).astype(np.float32)
        scale, _ = S.channel_scale([activations], 12)
        recipe = S.Recipe(n_features=48, topk_pct=0.1, seed=2)
        model = S.build(12, recipe, scale)

        path = S.save_inference_checkpoint(tmp_path / "sae.pt", model, recipe,
                                           extra={"step": 3000})
        frozen = C.FrozenSAE.from_torch_checkpoint(path)
        assert frozen.d_in == 12 and frozen.n_features == 48
        assert frozen.k == recipe.k(12)

        _, expected, _ = model(torch.from_numpy(activations))
        got = frozen.encode(activations, chunk=16).toarray()
        np.testing.assert_allclose(got, expected.detach().numpy(), rtol=0, atol=1e-5)

    def test_refuses_to_save_broken_parameters(self, tmp_path):
        model = S.build(8, S.Recipe(n_features=32), np.ones(8))
        with torch.no_grad():
            model.core.pre_bias[0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            S.save_inference_checkpoint(tmp_path / "bad.pt", model, S.Recipe(n_features=32))
