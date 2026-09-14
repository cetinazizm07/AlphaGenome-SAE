"""Ablation: the arithmetic, the controls, and a cluster-aware permutation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ag_sae import ablate as A


class TestAblateActivation:
    def test_removing_a_feature_removes_exactly_its_contribution(self):
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(6, 4))
        codes = np.zeros((10, 6), dtype=np.float32)
        codes[:, 2] = 1.5
        activation = rng.normal(size=(10, 4)).astype(np.float32)
        out = A.ablate_activation(activation, codes, directions, [2])
        assert out == pytest.approx(activation - 1.5 * directions[2], abs=1e-5)

    def test_a_silent_feature_changes_nothing(self):
        rng = np.random.default_rng(1)
        directions = rng.normal(size=(6, 4))
        codes = np.zeros((10, 6), dtype=np.float32)
        activation = rng.normal(size=(10, 4)).astype(np.float32)
        assert A.ablate_activation(activation, codes, directions, [3]) == pytest.approx(activation)

    def test_an_empty_feature_list_is_the_identity(self):
        activation = np.ones((4, 3), dtype=np.float32)
        out = A.ablate_activation(activation, np.ones((4, 2), dtype=np.float32),
                                  np.ones((2, 3)), [])
        assert out == pytest.approx(activation)

    def test_an_index_outside_the_dictionary_is_refused(self):
        with pytest.raises(ValueError, match="outside the dictionary"):
            A.ablate_activation(np.ones((2, 3), dtype=np.float32),
                                np.ones((2, 2), dtype=np.float32), np.ones((2, 3)), [5])

    def test_a_width_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="match the activation width"):
            A.ablate_activation(np.ones((2, 3), dtype=np.float32),
                                np.ones((2, 2), dtype=np.float32), np.ones((2, 9)), [0])


class TestControls:
    def test_controls_match_the_target_activity(self):
        activity = np.array([0.01, 0.02, 5.0, 5.1, 4.9, 0.03, 5.05])
        rng = np.random.default_rng(0)
        chosen = A.matched_controls(activity, [2], n_controls=2, rng=rng, tolerance=0.1)
        assert len(chosen) == 2
        assert all(abs(activity[c] - 5.0) <= 0.5 for c in chosen)
        assert 2 not in chosen

    def test_the_target_is_never_its_own_control(self):
        activity = np.arange(20.0)
        rng = np.random.default_rng(1)
        chosen = A.matched_controls(activity, [7, 8], n_controls=3, rng=rng)
        assert 7 not in chosen and 8 not in chosen

    def test_it_falls_back_when_nothing_matches(self):
        activity = np.array([0.0, 100.0, 200.0])
        rng = np.random.default_rng(2)
        chosen = A.matched_controls(activity, [0], n_controls=1, rng=rng, tolerance=0.01)
        assert len(chosen) == 1


class TestRunAblation:
    def _setup(self, n=200, d=6, h=8, seed=0):
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(h, d))
        labels = np.zeros(n, dtype=bool)
        labels[:40] = True
        codes = np.zeros((n, h), dtype=np.float32)
        codes[labels, 1] = 2.0          # feature 1 fires only at the concept
        codes[:, 5] = 0.5               # feature 5 fires everywhere
        activation = rng.normal(size=(n, d)).astype(np.float32)
        return directions, codes, labels, activation

    def test_a_concept_feature_is_selective(self):
        directions, codes, labels, activation = self._setup()
        result = A.run_ablation(lambda x: x, activation, codes, directions,
                                labels, [1], control_features=[5], name="PLS")
        assert result.summary["selectivity"] > 5
        # The control fires everywhere, so removing it is not selective at all.
        assert result.summary["control_selectivity"] == pytest.approx(1.0, abs=0.05)
        assert result.summary["selectivity_over_control"] > 4

    def test_a_feature_that_fires_everywhere_is_not_selective(self):
        directions, codes, labels, activation = self._setup()
        result = A.run_ablation(lambda x: x, activation, codes, directions, labels, [5])
        assert result.summary["selectivity"] == pytest.approx(1.0, abs=0.05)

    def test_per_site_rows_line_up_with_the_sites(self):
        directions, codes, labels, activation = self._setup()
        result = A.run_ablation(lambda x: x, activation, codes, directions, labels, [1])
        assert len(result.per_site) == len(labels)
        assert result.per_site.is_concept.tolist() == labels.tolist()

    def test_a_concept_with_no_negatives_is_refused(self):
        directions, codes, labels, activation = self._setup()
        with pytest.raises(ValueError, match="inside and outside"):
            A.run_ablation(lambda x: x, activation, codes, directions,
                           np.ones(len(labels), dtype=bool), [1])


class TestPermutation:
    def _clustered(self, n_clusters=12, per=20, seed=0):
        rng = np.random.default_rng(seed)
        clusters = np.repeat(np.arange(n_clusters), per)
        labels = np.zeros(clusters.size, dtype=bool)
        labels[clusters % 3 == 0] = True
        effect = rng.normal(size=clusters.size)
        return effect, labels, clusters

    def test_a_real_effect_is_detected(self):
        effect, labels, clusters = self._clustered()
        effect = effect + labels * 6.0
        p = A.paired_permutation_p(effect, labels, clusters, n_permutations=200, seed=0)
        assert p < 0.05

    def test_no_effect_gives_a_large_p(self):
        effect, labels, clusters = self._clustered()
        p = A.paired_permutation_p(effect, labels, clusters, n_permutations=200, seed=0)
        assert p > 0.05

    def test_clustering_is_more_conservative_than_ignoring_it(self):
        # The whole reason clusters exist. A difference driven by whole windows
        # must be easy to reproduce when windows are swapped, and hard when
        # sites are treated as independent.
        rng = np.random.default_rng(4)
        clusters = np.repeat(np.arange(10), 50)
        labels = clusters < 5
        # The effect is a property of the window, constant inside it. There are
        # really ten observations here, not five hundred.
        effect = np.repeat(rng.normal(size=10), 50) + rng.normal(0, 0.05, clusters.size)
        clustered = A.paired_permutation_p(effect, labels, clusters,
                                           n_permutations=400, seed=1)
        per_site = A.paired_permutation_p(effect, labels, None,
                                          n_permutations=400, seed=1)
        # Per-site: 0.003, convincing and wrong. Clustered: 0.34, which is the
        # honest reading of ten windows.
        assert per_site < 0.01
        assert clustered > 0.2

    def test_p_is_never_zero(self):
        effect, labels, clusters = self._clustered()
        p = A.paired_permutation_p(effect + labels * 50, labels, clusters,
                                   n_permutations=50, seed=0)
        assert p > 0

    def test_a_one_class_label_vector_is_refused(self):
        with pytest.raises(ValueError, match="inside and outside"):
            A.paired_permutation_p(np.ones(10), np.ones(10, dtype=bool))


class TestIntervention:
    """The hook mechanics, against a stand-in with the port's shapes."""

    def _model(self):
        import torch
        from torch import nn

        class Block(nn.ModuleDict):
            pass

        class Mha(nn.Module):
            def forward(self, x, *_a, **_k):
                return x * 0.0

        class Encoder(nn.Module):
            def forward(self, x):
                trunk = x.transpose(1, 2)
                return trunk, {"bin_size_4": x.transpose(1, 2).clone()}

        class Tower(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block({"mha": Mha()})])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.tower = Tower()

        return Model()

    def test_a_tower_hook_replaces_the_residual_stream(self):
        import torch
        from ag_sae.extract import TAPS

        model = self._model()
        seen = {}

        def replace(values):
            seen["shape"] = values.shape
            return values + 7.0

        stream = torch.zeros(1, 5, 3)
        with A.Intervention(model, TAPS["resid_pre_b0"], replace):
            captured = {}

            def spy(_m, args, kwargs):
                captured["x"] = args[0].clone()
                return args, kwargs

            handle = model.tower.blocks[0]["mha"].register_forward_pre_hook(
                spy, with_kwargs=True)
            model.tower.blocks[0]["mha"](stream)
            handle.remove()
        assert seen["shape"] == (5, 3)
        assert torch.allclose(captured["x"][0], torch.full((5, 3), 7.0))

    def test_an_encoder_hook_rewrites_the_skip_connection(self):
        import torch
        from ag_sae.extract import TAPS

        model = self._model()
        with A.Intervention(model, TAPS["bin_size_4"], lambda v: v + 3.0):
            _trunk, intermediates = model.encoder(torch.zeros(1, 6, 4))
        assert torch.allclose(intermediates["bin_size_4"],
                              torch.full_like(intermediates["bin_size_4"], 3.0))

    def test_the_hook_is_removed_on_exit(self):
        import torch
        from ag_sae.extract import TAPS

        model = self._model()
        with A.Intervention(model, TAPS["bin_size_4"], lambda v: v + 3.0):
            pass
        _trunk, intermediates = model.encoder(torch.zeros(1, 6, 4))
        assert torch.allclose(intermediates["bin_size_4"],
                              torch.zeros_like(intermediates["bin_size_4"]))


class TestFeatureRemover:
    def test_only_the_sampled_rows_are_touched(self):
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(4, 3))
        codes = np.zeros((2, 4), dtype=np.float32)
        codes[:, 0] = 1.0
        activation = np.ones((10, 3), dtype=np.float32)
        replace = A.feature_remover(codes, directions, [0], positions=np.array([2, 7]))
        out = replace(activation)
        expected = np.tile(1.0 - directions[0], (2, 1))
        assert out[[2, 7]] == pytest.approx(expected, abs=1e-5)
        untouched = [i for i in range(10) if i not in (2, 7)]
        assert out[untouched] == pytest.approx(np.ones((8, 3)))

    def test_a_position_count_mismatch_is_refused(self):
        replace = A.feature_remover(np.ones((2, 4), dtype=np.float32),
                                    np.ones((4, 3)), [0], positions=np.array([1]))
        with pytest.raises(ValueError, match="one entry per code row"):
            replace(np.ones((5, 3), dtype=np.float32))
