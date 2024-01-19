import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from codeless_ml.ml.transformer.attention import GlobalSelfAttention, CausalSelfAttention, CrossAttention


class TestAttention(unittest.TestCase):

    def test_gsa(self):
        d_model = 256
        batch = 32
        seq_len = 16
        emb = tf.random.normal([batch, seq_len, d_model])
        gsa = GlobalSelfAttention(num_heads=2, key_dim=d_model)
        output = gsa(emb)
        self.assertEqual(output.shape, (batch, seq_len, d_model))

    def test_ca(self):
        d_model = 256
        batch = 32
        seq_len = 16
        emb = tf.random.normal([batch, seq_len, d_model])
        context = tf.random.normal([batch, 8, d_model])
        ca = CrossAttention(num_heads=2, key_dim=d_model)
        output = ca(emb, context)
        self.assertEqual(output.shape, (batch, seq_len, d_model))

    def test_csa(self):
        d_model = 256
        batch = 32
        seq_len = 16
        emb = tf.random.normal([batch, seq_len, d_model])
        csa = CausalSelfAttention(num_heads=2, key_dim=d_model)
        out1 = csa(emb[:, :4, :])
        out2 = csa(emb)[:, :4, :]
        npt.assert_allclose(out1, out2, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
