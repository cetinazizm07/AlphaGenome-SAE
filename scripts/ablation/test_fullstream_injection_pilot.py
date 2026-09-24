"""CPU contracts for the dev-calibrated injection tensor construction."""

import unittest

import torch

from run_cloud_fullstream_injection_pilot import reconstructed_injection_triplet


class FakeCore:
    def encode(self, value):
        return value, {}

    def activation(self, value):
        return value

    def get_sparse_activations(self, value):
        return value

    def decode(self, codes, _params):
        return codes


class FakeSAE:
    d_in = 2
    hidden = 2
    channel_scale = torch.ones(2)
    core = FakeCore()


class InjectionTests(unittest.TestCase):
    def test_same_reconstruction_and_one_feature_changed_per_arm(self):
        activation = torch.tensor([[[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]]])
        baseline, target, control, counts = reconstructed_injection_triplet(
            FakeSAE(), activation, 0, 1, [1, 2], 5.0, 7.0, batch_size=1
        )
        torch.testing.assert_close(baseline, activation)
        torch.testing.assert_close(target[0, 0], activation[0, 0])
        torch.testing.assert_close(control[0, 0], activation[0, 0])
        torch.testing.assert_close(target[0, 1:], torch.tensor([[5.0, 3.0], [5.0, 0.0]]))
        torch.testing.assert_close(control[0, 1:], torch.tensor([[0.0, 7.0], [4.0, 7.0]]))
        self.assertEqual(counts["target_was_zero"], 1)
        self.assertEqual(counts["control_was_zero"], 1)

    def test_rejects_duplicate_positions(self):
        activation = torch.ones(1, 3, 2)
        with self.assertRaisesRegex(ValueError, "positions"):
            reconstructed_injection_triplet(FakeSAE(), activation, 0, 1, [1, 1], 5.0, 7.0)


if __name__ == "__main__":
    unittest.main()
