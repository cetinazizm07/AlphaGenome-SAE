"""Small executable contracts for the deployed full-stream pilot."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from run_cloud_fullstream_control_pilot import FullStreamTower, select_control


class TinyTower:
    def __init__(self):
        self.blocks = [object()]

    def _forward_block(self, block, x, pair_x, compute_dtype):
        assert block is self.blocks[0]
        # Both branches must receive the edited stream, not just MHA.
        pair_x = x.sum(dim=-1)
        return x + 2 * x, pair_x


class FullStreamTests(unittest.TestCase):
    def test_edits_pair_and_residual_and_restores_method(self):
        tower = TinyTower()
        model = SimpleNamespace(tower=tower)
        tap = SimpleNamespace(kind="tower", key=0)
        x = torch.tensor([[[1.0, 2.0]]])
        with FullStreamTower(model, tap, lambda value: value + 1) as editor:
            changed, pair = tower._forward_block(tower.blocks[0], x, None, None)
        torch.testing.assert_close(changed, torch.tensor([[[6.0, 9.0]]]))
        torch.testing.assert_close(pair, torch.tensor([[5.0]]))
        self.assertEqual(editor.calls, 1)
        self.assertNotIn("_forward_block", vars(tower))
        original, original_pair = tower._forward_block(tower.blocks[0], x, None, None)
        torch.testing.assert_close(original, torch.tensor([[[3.0, 6.0]]]))
        torch.testing.assert_close(original_pair, torch.tensor([[3.0]]))

    def test_control_rule_is_deterministic_and_excludes_high_auroc(self):
        firing = np.array([0.999, 0.783, 0.784, 0.999], dtype=np.float32)
        auroc = np.array([[0.91], [0.53], [0.54], [0.90]], dtype=np.float32)
        first, candidates = select_control(0, firing, auroc)
        second, again = select_control(0, firing, auroc)
        self.assertEqual(candidates, [1, 2])
        self.assertEqual(again, candidates)
        self.assertEqual(first, second)
        self.assertIn(first, candidates)


if __name__ == "__main__":
    unittest.main()
