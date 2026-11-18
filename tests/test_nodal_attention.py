import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from model_utils import attentive_node_features


class NodalAttentionTests(unittest.TestCase):
    def setUp(self):
        self.layer = attentive_node_features(hidden_size=2)
        with torch.no_grad():
            self.layer.transform.weight.zero_()
            self.layer.transform.bias.zero_()
        self.features = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [50.0, 50.0]],
             [[2.0, 0.0], [0.0, 2.0], [50.0, 50.0], [50.0, 50.0]]],
            requires_grad=True,
        )
        self.lengths = [3, 2]

    def run_and_capture_attention(self, mode):
        captured = []
        original_softmax = functional.softmax

        def capture_softmax(scores, dim):
            weights = original_softmax(scores, dim=dim)
            captured.append(weights)
            return weights

        with patch('model_utils.F.softmax', side_effect=capture_softmax):
            output = self.layer(self.features, self.lengths, mode)
        return output, captured[0]

    def test_none_returns_features_unchanged(self):
        output = self.layer(self.features, self.lengths, None)
        self.assertIs(output, self.features)

    def test_past_masks_future_and_padding_and_includes_self(self):
        _, attention = self.run_and_capture_attention('past')
        for batch_index, length in enumerate(self.lengths):
            for query_index in range(length):
                self.assertTrue(torch.all(attention[batch_index, query_index, query_index + 1:] == 0))
                self.assertGreater(attention[batch_index, query_index, query_index].item(), 0)
                self.assertAlmostEqual(attention[batch_index, query_index].sum().item(), 1.0)
            self.assertTrue(torch.all(attention[batch_index, :, length:] == 0))

    def test_global_uses_all_valid_utterances_and_masks_padding(self):
        _, attention = self.run_and_capture_attention('global')
        for batch_index, length in enumerate(self.lengths):
            self.assertTrue(torch.all(attention[batch_index, :, length:] == 0))
            self.assertTrue(torch.allclose(
                attention[batch_index, 0, :length],
                torch.full((length,), 1.0 / length),
            ))
            self.assertTrue(torch.allclose(
                attention[batch_index, 0].sum(), torch.tensor(1.0)
            ))

    def test_past_output_is_independent_of_future_features(self):
        baseline = self.layer(self.features, self.lengths, 'past')
        changed_features = self.features.detach().clone()
        changed_features[0, 2] = torch.tensor([-100.0, 100.0])
        changed = self.layer(changed_features, self.lengths, 'past')
        self.assertTrue(torch.allclose(baseline[0, :2], changed[0, :2]))

    def test_forward_and_backward_are_finite(self):
        output = self.layer(self.features, self.lengths, 'past')
        output.sum().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(self.features.grad)
        self.assertTrue(torch.isfinite(self.features.grad).all())


if __name__ == '__main__':
    unittest.main()
