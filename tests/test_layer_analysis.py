import unittest
from types import SimpleNamespace

import numpy as np
import torch

from analyze_layer_cka import linear_cka
from model import DAGERC_multimodal


class LayerAnalysisTests(unittest.TestCase):
    def test_linear_cka_is_symmetric_and_self_similarity_is_one(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(12, 6))
        y = rng.normal(size=(12, 4))
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=10)
        self.assertAlmostEqual(linear_cka(x, y), linear_cka(y, x), places=10)

    def test_constant_features_produce_nan_cka(self):
        x = np.ones((8, 3))
        self.assertTrue(np.isnan(linear_cka(x, x)))

    def test_optional_intermediates_preserve_normal_predictions(self):
        args = SimpleNamespace(
            use_text=True, use_audio=True, dropout=0.0, gnn_layers=4,
            hidden_dim=3, no_rel_attn=False, attn_type='rgcn',
            nodal_att_type=None, mlp_layers=2, text_dim=6, audio_dim=7,
            fusion_dim_D=4, fusion_dim_O=8, single_modal_dim=8,
            simple_nn=False, standard_gnn=False, edl_r2=False, windowp=1,
        )
        model = DAGERC_multimodal(args, num_class=5).eval()
        features_text = torch.randn(1, 3, 6)
        features_audio = torch.randn(1, 3, 7)
        adj = torch.zeros(1, 3, 3)
        s_mask = torch.ones(1, 3, 3, dtype=torch.long)
        s_mask_onehot = torch.zeros(1, 3, 3, 2)
        lengths = torch.tensor([3])

        with torch.inference_mode():
            baseline = model(features_text, features_audio, adj, s_mask, s_mask_onehot, lengths)
            predictions, intermediate = model(
                features_text, features_audio, adj, s_mask, s_mask_onehot, lengths,
                return_intermediates=True,
            )

        self.assertTrue(torch.equal(baseline, predictions))
        for layer in [f'H{i}' for i in range(5)]:
            self.assertEqual(tuple(intermediate[layer].shape), (1, 3, 3))
        self.assertEqual(tuple(intermediate['Fused'].shape), (1, 3, 8))
        self.assertEqual(tuple(intermediate['pre_mlp'].shape), (1, 3, 23))


if __name__ == '__main__':
    unittest.main()
