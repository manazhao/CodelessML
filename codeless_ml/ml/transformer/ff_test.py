import unittest
import tensorflow as tf
import numpy as np

from codeless_ml.ml.transformer.ff import FeedForward


class TestFeedForward(unittest.TestCase):

    def test_call(self):
        d_model = 256
        dff = 2048
        ffn = FeedForward(d_model, dff)
        batch = 32
        seq_len = 16
        emb = tf.random.normal([batch, seq_len, d_model])
        output = ffn(emb)
        self.assertEqual(output.shape, (batch, seq_len, d_model))


if __name__ == '__main__':
    unittest.main()
