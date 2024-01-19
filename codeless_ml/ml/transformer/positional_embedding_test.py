import os
import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from codeless_ml.ml.transformer.positional_embedding import PositionalEmbedding


class TestPositionalEmbedding(unittest.TestCase):

    def test_initialization(self):
        vocab_size = 1000
        d_model = 512
        max_length = 50

        pe = PositionalEmbedding(vocab_size, d_model, max_length)

        # Check attributes
        self.assertEqual(pe.vocab_size, vocab_size)
        self.assertEqual(pe.d_model, d_model)
        self.assertEqual(pe.max_length, max_length)

        self.assertEqual(pe.positional_embedding.shape, (max_length, d_model))

    def test_call(self):
        vocab_size = 10
        d_model = 4
        max_length = 6
        input_tensor = tf.constant([[1, 4, 2, 0, 0]])  # Input with padding

        pe = PositionalEmbedding(vocab_size, d_model, max_length)
        output = pe(input_tensor)

        # Check output shape
        input_shape = tf.shape(input_tensor)
        self.assertEqual(
            output.shape,
            (input_shape[0], input_shape[1], d_model))  # Padding removed
        # sanity check the pos embedding vectors.
        vec1 = pe.positional_embedding[1, :]
        vec2 = pe.positional_embedding[2, :]
        vec5 = pe.positional_embedding[5, :]
        similarity1 = np.dot(vec1 / np.linalg.norm(vec1),
                             vec2 / np.linalg.norm(vec2))
        similarity2 = np.dot(vec1 / np.linalg.norm(vec1),
                             vec5 / np.linalg.norm(vec5))
        self.assertGreater(similarity1, similarity2)

    def test_mask(self):
        vocab_size = 10
        d_model = 4
        max_length = 6
        input_tensor = tf.constant([[1, 4, 0, 0, 2]])  # Input with padding
        pe = PositionalEmbedding(vocab_size, d_model, max_length)
        emb = pe(input_tensor)
        npt.assert_array_equal(emb._keras_mask,
                               [[True, True, False, False, True]])

    def _get_temp_dir(self, path: str):
        return os.path.join(os.environ["TEST_TMPDIR"], path)

    def test_save_and_load(self):
        vocab_size = 10
        d_model = 4
        max_length = 6
        layer_name = "positional_embedding_layer"
        pe = PositionalEmbedding(vocab_size,
                                 d_model,
                                 max_length,
                                 name=layer_name)
        input_tensor = tf.constant([[1, 4, 0, 0, 2]])  # Input with padding
        pe = PositionalEmbedding(vocab_size, d_model, max_length)
        emb = pe(input_tensor)

        # add the pe layer into a sequential model
        config = pe.get_config()
        weights = pe.get_weights()

        reloaded_pe = PositionalEmbedding.from_config(config)
        # Note: need to call the layer to initialize the weights...
        _ = reloaded_pe(input_tensor)
        reloaded_pe.set_weights(weights)
        npt.assert_allclose(reloaded_pe(input_tensor), emb)


if __name__ == '__main__':
    unittest.main()
