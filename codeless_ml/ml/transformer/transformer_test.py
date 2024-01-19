import unittest
import tensorflow as tf
import numpy as np

from codeless_ml.ml.transformer.transformer import Transformer


class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        num_layers = 4
        d_model = 128
        num_heads = 8
        dropout_rate = 0.1
        dff = 512
        input_vocab_size = 10
        target_vocab_size = 20

        transformer = Transformer(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  input_vocab_size=input_vocab_size,
                                  target_vocab_size=target_vocab_size,
                                  dropout_rate=dropout_rate)

        batch = 32
        input_seq_len = 16
        target_seq_len = 8
        input_seq = tf.cast(np.random.randint(0,
                                              input_vocab_size,
                                              size=(batch, input_seq_len)),
                            dtype=tf.int32)
        target_seq = tf.cast(np.random.randint(0,
                                               target_vocab_size,
                                               size=(batch, target_seq_len)),
                             dtype=tf.int32)
        inputs = (input_seq, target_seq)
        logits = transformer(inputs)
        self.assertEqual(logits.shape,
                         [batch, target_seq_len, target_vocab_size])
        last_attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
        self.assertEqual(last_attn_scores.shape,
                         (batch, num_heads, target_seq_len, input_seq_len))

    def test_mask(self):
        num_layers = 4
        d_model = 128
        num_heads = 8
        dropout_rate = 0.1
        dff = 512
        input_vocab_size = 10
        target_vocab_size = 20

        transformer = Transformer(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  input_vocab_size=input_vocab_size,
                                  target_vocab_size=target_vocab_size,
                                  dropout_rate=dropout_rate)

        input_seq = tf.constant([[1, 2, 3, 4]])
        target_seq = tf.constant([[2, 3, 4, 0]])
        inputs = (input_seq, target_seq)
        logits = transformer(inputs)
        np.testing.assert_array_equal(logits._keras_mask,
                                      [[True, True, True, False]])

        labels = target_seq
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none")
        loss = loss_fn(labels, logits)
        # now set the values of the masked logits.
        mutated_logits = logits
        tf.tensor_scatter_nd_update(mutated_logits, [[0, 3]],
                                    tf.random.normal([1, target_vocab_size]))
        np.testing.assert_array_equal(loss_fn(labels, mutated_logits), loss)


if __name__ == '__main__':
    unittest.main()
