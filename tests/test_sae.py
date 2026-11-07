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

    def test_saved_model_reloads_for_sparse_match(self, tmp_path):
        recipe = S.Recipe(n_features=16, topk_pct=0.25, seed=3)
        model = S.build(4, recipe, np.ones(4, dtype=np.float32))
        path = S.save_inference_checkpoint(tmp_path / "sae.pt", model, recipe)
        loaded = S.BorzoiSAE.from_checkpoint(path)

        raw = torch.randn(3, 4)
        expected = model(raw)[1]
        values, indices = loaded.encode_topk(raw)
        actual = torch.zeros_like(expected).scatter_(-1, indices, values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_refuses_to_save_broken_parameters(self, tmp_path):
        model = S.build(8, S.Recipe(n_features=32), np.ones(8))
        with torch.no_grad():
            model.core.pre_bias[0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            S.save_inference_checkpoint(tmp_path / "bad.pt", model, S.Recipe(n_features=32))


class TestRecipeV2:
    def test_legacy_is_the_first_campaign(self):
        legacy = S.Recipe.legacy()
        assert (legacy.lr, legacy.steps, legacy.lr_schedule, legacy.init) == (1e-5, 6000, "constant", "default")
        assert not legacy.uses_aux and legacy.seed_scope == "global" and legacy.shuffle == "shard"
        assert legacy.learning_rate(0) == legacy.learning_rate(5999) == 1e-5

    def test_warmup_then_cosine_to_the_floor(self):
        r = S.Recipe(lr=1e-3, steps=1000, lr_warmup_steps=100, lr_final_ratio=0.1)
        assert r.learning_rate(0) == pytest.approx(1e-5)
        assert r.learning_rate(99) == pytest.approx(1e-3)
        assert r.learning_rate(550) == pytest.approx(0.55e-3)
        assert r.learning_rate(1000) == pytest.approx(1e-4)

    def test_explicit_k_overrides_the_percentage(self):
        assert S.Recipe(n_features=8192, topk=64).k(1536) == 64

    def test_seeds_differ_by_tap_and_data_but_not_in_legacy(self):
        r = S.Recipe(seed=1)
        a = S.derive_seed(r, "resid_pre_b0", "foldA", "init")
        assert a == S.derive_seed(r, "resid_pre_b0", "foldA", "init")
        assert len({a, S.derive_seed(r, "resid_pre_b4", "foldA", "init"),
                    S.derive_seed(r, "resid_pre_b0", "foldB", "init"),
                    S.derive_seed(r, "resid_pre_b0", "foldA", "data")}) == 4
        assert S.derive_seed(S.Recipe.legacy(seed=1), "resid_pre_b4", "x", "init") == 1


class TestDeadLatentDefences:
    def model(self):
        return S.build(12, S.Recipe(n_features=48, topk=4), np.ones(12), seed=0)

    def test_tied_init_has_unit_decoder_and_transposed_encoder(self):
        m = self.model()
        dec = m.core.decoder.weight
        torch.testing.assert_close(dec.norm(dim=0), torch.ones(48))
        torch.testing.assert_close(m.core.encoder.weight, dec.T)

    def test_projected_gradient_keeps_columns_on_the_sphere_to_first_order(self):
        m = self.model()
        m.core.decoder.weight.grad = torch.randn_like(m.core.decoder.weight)
        m.project_decoder_grad_()
        radial = (m.core.decoder.weight.grad * m.core.decoder.weight).sum(0)
        torch.testing.assert_close(radial, torch.zeros(48), atol=1e-5, rtol=0)

    def test_aux_uses_only_dead_latents_and_is_absent_without_them(self):
        m = self.model()
        raw = torch.randn(10, 12)
        *_, aux = m.forward_train(raw, torch.zeros(48, dtype=torch.bool), aux_k=8)
        assert aux is None
        dead = torch.zeros(48, dtype=torch.bool)
        dead[:5] = True
        recon, codes, scaled, aux = m.forward_train(raw, dead, aux_k=8)
        assert aux.shape == raw.shape
        # Only dead columns contribute: zeroing them silences aux.
        with torch.no_grad():
            m.core.decoder.weight[:, :5] = 0
        *_, aux_zero = m.forward_train(raw, dead, aux_k=8)
        torch.testing.assert_close(aux_zero, torch.zeros_like(aux_zero))
        # The main path is the unmodified forward.
        torch.testing.assert_close(m(raw)[0], m.forward_train(raw)[0])

    def test_mean_pre_bias(self):
        m = self.model()
        sample = torch.randn(256, 12) * 2 + 1
        m.init_pre_bias_(sample)
        from ag_sae.vendor.sae_borzoi import LN
        torch.testing.assert_close(m.core.pre_bias, LN(sample)[0].mean(0))
