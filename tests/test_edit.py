"""Batched residual edits must reproduce the first campaign's one-pass-per-condition results."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ag_sae import edit as E
from ag_sae import tm
from ag_sae.extract import one_hot
from ag_sae.intervene import FullStreamTower, run_replacement
from ag_sae.sae import BorzoiSAE

from tests.fake_alphagenome import FakeAlphaGenome

L = 128 * 32          # 32 bins
DIM = 16
BLOCK = 2


class _Tap:                       # what FullStreamTower reads from a TAPS entry
    kind, key, bin_bp = "tower", BLOCK, 128


@pytest.fixture(scope="module")
def model():
    return FakeAlphaGenome(dim=DIM, n_blocks=4).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def sae():
    torch.manual_seed(3)
    return BorzoiSAE(DIM, 64, 6, torch.linspace(1, 2, DIM)).eval().requires_grad_(False)


def random_sequences(n, seed=0):
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACGT"), L)) for _ in range(n)]
    return torch.stack([torch.from_numpy(one_hot(s)) for s in seqs])


def legacy_capture(model, x):
    """The first campaign's capture: a full forward with an identity editor."""
    captured = []

    def keep(v):
        captured.append(np.array(v, dtype=np.float32, copy=True))
        return v

    with torch.inference_mode(), FullStreamTower(model, _Tap, keep):
        model.predict(x, 0, heads=("atac",))
    return torch.from_numpy(captured[0])[None]


def legacy_predict(model, x, activation, heads):
    with torch.inference_mode(), FullStreamTower(model, _Tap, run_replacement(activation)):
        return model.predict(x, 0, heads=heads)


class TestCaptureResidual:
    def test_matches_a_full_forward_capture(self, model):
        x = random_sequences(2)
        stream = E.capture_residual(model, BLOCK, x)
        for i in range(2):
            torch.testing.assert_close(stream[i:i + 1], legacy_capture(model, x[i:i + 1]))

    def test_restores_the_model(self, model):
        x = random_sequences(1)
        before = model.predict(x, 0, heads=("atac",))["atac"][128]
        E.capture_residual(model, BLOCK, x)
        after = model.predict(x, 0, heads=("atac",))["atac"][128]
        torch.testing.assert_close(before, after, rtol=0, atol=0)


class TestRunConditions:
    heads = ("atac", "chip_tf")

    def legacy_results(self, model, sae, x, bins, feature):
        """Original, reconstruction and one ablation, one pass each."""
        act = legacy_capture(model, x)
        rows = act[0, bins].float()
        codes, params = sae.encode_codes(rows)
        recon = act.clone()
        recon[0, bins] = sae.decode_codes(codes, params)
        ablated_codes = codes.clone()
        ablated_codes[:, feature] = 0
        abl = act.clone()
        abl[0, bins] = sae.decode_codes(ablated_codes, params)
        out = {"original": model.predict(x, 0, heads=self.heads)}
        out["recon"] = legacy_predict(model, x, recon, self.heads)
        out["ablate"] = legacy_predict(model, x, abl, self.heads)
        return {k: {h: v[h][128][0].mean(-1).numpy() for h in self.heads} for k, v in out.items()}

    @pytest.mark.parametrize("max_batch", [1, 2, 3])
    def test_batched_equals_one_pass_per_condition(self, model, sae, max_batch):
        x = random_sequences(1, seed=5)
        bins = [3, 4, 10, 20]
        feature = int(sae.encode_codes(legacy_capture(model, x)[0, bins])[0].sum(0).argmax())
        expected = self.legacy_results(model, sae, x, bins, feature)
        conditions = [E.Condition("original"),
                      E.Condition("recon", 0, bins, E.sae_edit(sae, "recon")),
                      E.Condition("ablate", 0, bins,
                                  E.sae_edit(sae, "ablate", E.zero_feature(feature)))]
        got = E.run_conditions(model, BLOCK, x, conditions, E.track_mean_profile(self.heads),
                               heads=self.heads, max_batch=max_batch)
        for name in expected:
            for head in self.heads:
                np.testing.assert_allclose(got[name][head], expected[name][head],
                                           rtol=1e-5, atol=1e-6)
        assert not np.allclose(got["ablate"]["atac"], got["recon"]["atac"])

    def test_uses_fewer_forwards(self, model, sae):
        x = random_sequences(1)
        conditions = [E.Condition("original")] + [
            E.Condition(f"c{i}", 0, [i], E.sae_edit(sae, f"c{i}")) for i in range(5)]
        model.calls = 0
        E.run_conditions(model, BLOCK, x, conditions, E.track_mean_profile(self.heads),
                         heads=self.heads, max_batch=6)
        assert model.calls == 1

    def test_reference_to_a_later_condition_fails_loudly(self, model, sae):
        x = random_sequences(2)
        conditions = [E.Condition("rescue", 1, [2], E.sae_edit(sae, "r", E.set_feature_from(0, "wt"))),
                      E.Condition("wt", 0, [2], E.sae_edit(sae, "wt"))]
        with pytest.raises(KeyError, match="must run before"):
            E.run_conditions(model, BLOCK, x, conditions, E.track_mean_profile(self.heads),
                             heads=self.heads)

    def test_condition_needs_bins_and_edit_together(self):
        with pytest.raises(ValueError):
            E.Condition("x", 0, [1])
        with pytest.raises(ValueError):
            E.Condition("x", 0, [], lambda r, m: r)


class TestFactorialModel:
    heads = ("chip_tf", "atac")

    def legacy_factorial(self, model, sae, wt, mut, anchor, feature):
        """The first campaign's six-pass 2x2, transcribed from motif_intervene."""
        pos = [anchor]
        raw_wt, raw_mut = legacy_capture(model, wt), legacy_capture(model, mut)
        wt_codes, p_wt = sae.encode_codes(raw_wt[0, pos])
        mut_codes, p_mut = sae.encode_codes(raw_mut[0, pos])

        def with_rows(raw, codes, params):
            act = raw.clone()
            act[0, pos] = sae.decode_codes(codes, params)
            return act

        abl = wt_codes.clone()
        abl[:, feature] = 0
        res = mut_codes.clone()
        res[:, feature] = wt_codes[:, feature]
        runs = {"wt": (wt, with_rows(raw_wt, wt_codes, p_wt)),
                "ism": (mut, with_rows(raw_mut, mut_codes, p_mut)),
                "ablation": (wt, with_rows(raw_wt, abl, p_wt)),
                "rescue": (mut, with_rows(raw_mut, res, p_mut))}
        out = []
        for name in tm.FACTORIAL_CONDITIONS:
            seq, act = runs[name]
            pred = legacy_predict(model, seq, act, self.heads)
            window = slice(max(0, anchor - 1), anchor + 2)
            out.append([pred[h][128][0, window].mean().item() for h in self.heads])
        return np.array(out), (wt_codes[0, feature].item(), mut_codes[0, feature].item())

    def test_matches_the_six_pass_protocol_through_tangermeme_predict(self, model, sae):
        from tangermeme.predict import predict

        wt = random_sequences(2, seed=11)
        mut = wt.clone()
        mut[:, 5 * 128:5 * 128 + 19] = mut[:, 5 * 128:5 * 128 + 19].flip(1)
        anchors = torch.tensor([5.0, 5.0])
        feature = int(sae.encode_codes(legacy_capture(model, wt[:1])[0, [5]])[0].argmax())
        wrapper = tm.FactorialMotifModel(model, sae, BLOCK, feature, self.heads, max_batch=2)
        readouts, latents = predict(wrapper, tm.to_tangermeme(wt), args=(tm.to_tangermeme(mut), anchors),
                                    batch_size=1, device="cpu")
        assert readouts.shape == (2, 4, 2) and latents.shape == (2, 2)
        for i in range(2):
            expected, (wt_value, mut_value) = self.legacy_factorial(
                model, sae, wt[i:i + 1], mut[i:i + 1], 5, feature)
            np.testing.assert_allclose(readouts[i].numpy(), expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(latents[i].numpy(), [wt_value, mut_value], rtol=1e-5, atol=1e-6)


class TestSAEFeatureModel:
    def test_codes_match_capture_then_encode(self, model, sae):
        x = random_sequences(2, seed=2)
        wrapper = tm.SAEFeatureModel(model, sae, BLOCK, [0, 5, 7])
        got = wrapper(tm.to_tangermeme(x))
        assert got.shape == (2, L // 128, 3)
        for i in range(2):
            codes, _ = sae.encode_codes(legacy_capture(model, x[i:i + 1])[0])
            torch.testing.assert_close(got[i], codes[:, [0, 5, 7]], rtol=1e-5, atol=1e-6)


class TestMotifTools:
    def test_shuffle_keeps_composition_and_changes_the_site(self):
        x = tm.to_tangermeme(random_sequences(1, seed=4))
        y = tm.shuffle_site(x, 100, 119, seed=0)
        assert not torch.equal(x[..., 100:119], y[..., 100:119])
        torch.testing.assert_close(x[..., 100:119].sum(-1), y[..., 100:119].sum(-1))
        torch.testing.assert_close(x[..., :100], y[..., :100])
        torch.testing.assert_close(x[..., 119:], y[..., 119:])

    def test_fimo_finds_an_implanted_ctcf_site(self):
        from ag_sae.motif_intervene import get_canonical_ctcf_motif

        motif = get_canonical_ctcf_motif(None)
        rng = np.random.default_rng(0)
        background = "".join(rng.choice(list("ACGT"), 4000))
        site = motif.consensus
        seq = background[:1500] + site + background[1500 + len(site):]
        x = tm.to_tangermeme(torch.from_numpy(one_hot(seq)))
        hits = tm.strongest_sites(x, motif, name="CTCF", threshold=1e-4)
        assert len(hits) == 1
        assert hits.loc[0, "start"] == 1500 and hits.loc[0, "end"] == 1500 + len(site)


class TestAutocastIsolation:
    def test_sae_edits_run_in_float32_under_bf16_autocast(self, sae):
        model = FakeAlphaGenome(dim=DIM, n_blocks=4, autocast=True).eval().requires_grad_(False)
        x = random_sequences(1, seed=8)
        bins = list(range(32))
        memo = {}
        E.run_conditions(model, BLOCK, x, [E.Condition("recon", 0, bins, E.sae_edit(sae, "recon"))],
                         E.track_mean_profile(("atac",)), heads=("atac",), memo=memo)
        stream = E.capture_residual(model, BLOCK, x)[0].float()
        expected, _ = sae.encode_codes(stream)             # fp32, outside any autocast
        assert memo["recon"].dtype == torch.float32
        torch.testing.assert_close(memo["recon"], expected, rtol=1e-6, atol=1e-6)
