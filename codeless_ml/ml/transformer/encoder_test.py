import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from codeless_ml.ml.transformer.encoder import EncoderLayer, Encoder
from codeless_ml.ml.transformer.positional_embedding import PositionalEmbedding


class TestEncoder(unittest.TestCase):

    def test_encoder_layer(self):
        d_model = 256
        batch = 32
        seq_len = 16
        vocab_size = 10
        d_model = 512
        dff = 2048
        encoder = EncoderLayer(d_model=d_model,
                               num_heads=8,
                               dff=dff,
                               dropout_rate=0.1)
        emb = tf.random.normal([batch, seq_len, d_model])
        output = encoder(emb)
        self.assertEqual(output.shape, [batch, seq_len, d_model])

    def test_encoder(self):
        batch = 32
        seq_len = 16
        vocab_size = 10
        d_model = 512
        dff = 2048
        encoder = Encoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=vocab_size)
        input = tf.cast(np.random.randint(0, vocab_size,
                                          size=(batch, seq_len)),
                        dtype=tf.int32)
        output = encoder(input)
        self.assertEqual(output.shape, [batch, seq_len, d_model])

    def test_mask_for_encoder_layer(self):
        d_model = 4
        encoder_layer = EncoderLayer(d_model=d_model,
                                     num_heads=8,
                                     dff=16,
                                     dropout_rate=0.1)
        input = tf.cast([[1, 0, 0, 2]], dtype=tf.int32)
        pe = PositionalEmbedding(vocab_size=10, d_model=d_model)
        emb = pe(input)
        expected_mask = [[True, False, False, True]]
        npt.assert_array_equal(emb._keras_mask, expected_mask)
        dropout = tf.keras.layers.Dropout(0.1)
        emb_dropout = dropout(emb)
        npt.assert_array_equal(emb_dropout._keras_mask, expected_mask)
        output = encoder_layer(emb_dropout)
        npt.assert_array_equal(output._keras_mask, expected_mask)

    def test_mask_for_encoder(self):
        seq_len = 4
        vocab_size = 10
        d_model = 16
        dff = 32
        encoder = Encoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=vocab_size)
        input = tf.cast([[1, 1, 0, 0]], dtype=tf.int32)
        output = encoder(input)
        npt.assert_array_equal(output._keras_mask,
                               [[True, True, False, False]])

    def test_save_and_load(self):
        seq_len = 4
        vocab_size = 10
        d_model = 16
        dff = 32
        encoder = Encoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=vocab_size)
        input = tf.cast([[1, 1, 0, 0]], dtype=tf.int32)
        output = encoder(input)
        config = encoder.get_config()
        weights = encoder.get_weights()

        reloaded_encoder = Encoder.from_config(config)
        # note: create the weights by calling the layer once.
        _ = reloaded_encoder(input)
        reloaded_encoder.set_weights(weights)
        npt.assert_allclose(reloaded_encoder(input), output)

    def test_embedding_input(self):
        batch = 5
        seq_len = 4
        d_model = 16
        dff = 32
        # create an encoder without positional embedding layer.
        encoder = Encoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=None)
        x = tf.random.normal([batch, seq_len, d_model])
        output = encoder(x)
        self.assertEqual(output.shape, (batch, seq_len, d_model))


if __name__ == '__main__':
    unittest.main()
