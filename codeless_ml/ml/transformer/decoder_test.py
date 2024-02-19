import unittest
import tensorflow as tf
import numpy as np
import numpy.testing as npt

from codeless_ml.ml.transformer.decoder import DecoderLayer, Decoder
from codeless_ml.ml.transformer.positional_embedding import PositionalEmbedding


class TestDecoder(unittest.TestCase):

    def test_decoder_layer(self):
        d_model = 256
        batch = 32
        seq_len = 16
        target_seq = 8
        vocab_size = 10
        d_model = 512
        dff = 2048
        decoder = DecoderLayer(d_model=d_model,
                               num_heads=8,
                               dff=dff,
                               dropout_rate=0.1)
        emb = tf.random.normal([batch, seq_len, d_model])
        context_emb = tf.random.normal([batch, target_seq, d_model])
        output = decoder(emb, context_emb)
        self.assertEqual(output.shape, [batch, seq_len, d_model])

    def test_decoder(self):
        d_model = 256
        batch = 32
        seq_len = 16
        vocab_size = 10
        d_model = 512
        dff = 2048
        decoder = Decoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=vocab_size)
        input = tf.cast(np.random.randint(0, vocab_size,
                                          size=(batch, seq_len)),
                        dtype=tf.int32)
        target_seq = 8
        context_emb = tf.random.normal([batch, target_seq, d_model])
        output = decoder(input, context=context_emb)
        self.assertEqual(output.shape, [batch, seq_len, d_model])

    def test_mask(self):
        d_model = 4
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=8,
                                     dff=16,
                                     dropout_rate=0.1)
        input = tf.cast([[1, 0, 0, 2]], dtype=tf.int32)
        pe = PositionalEmbedding(vocab_size=10, d_model=d_model)
        context_emb = pe(input)
        target_seq = tf.cast([[1, 0]], dtype=tf.int32)
        npt.assert_array_equal(context_emb._keras_mask,
                               [[True, False, False, True]])
        target_emb = pe(target_seq)
        npt.assert_array_equal(target_emb._keras_mask, [[True, False]])
        output = decoder_layer(target_emb, context_emb)
        npt.assert_array_equal(output._keras_mask, [[True, False]])

        # now mutate the values of the masked positions in the context_emb and
        # it should lead the same result as the original context_emb.
        mutated_context_emb = context_emb
        indices = [[0, 1], [0, 2]]
        new_values = tf.random.normal([2, d_model])
        tf.tensor_scatter_nd_update(mutated_context_emb, indices, new_values)
        new_output = decoder_layer(target_emb, mutated_context_emb)
        npt.assert_array_equal(new_output, output)

    def test_save_and_load(self):
        d_model = 256
        batch = 32
        seq_len = 16
        vocab_size = 10
        d_model = 512
        dff = 2048
        decoder = Decoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=vocab_size)
        input = tf.cast(np.random.randint(0, vocab_size,
                                          size=(batch, seq_len)),
                        dtype=tf.int32)
        target_seq = 8
        context_emb = tf.random.normal([batch, target_seq, d_model])
        output = decoder(input, context=context_emb)

        # save and load
        config = decoder.get_config()
        weights = decoder.get_weights()

        reloaded_decoder = Decoder.from_config(config)
        _ = reloaded_decoder(input, context=context_emb)
        reloaded_decoder.set_weights(weights)
        npt.assert_allclose(reloaded_decoder(input, context=context_emb),
                            output)

    def test_embedding_input(self):
        batch = 5
        seq_len = 4
        d_model = 16
        dff = 32
        decoder = Decoder(num_layers=4,
                          d_model=d_model,
                          num_heads=8,
                          dff=dff,
                          vocab_size=None)
        input = tf.random.normal([batch, seq_len, d_model])
        target_seq = 8
        context_emb = tf.random.normal([batch, seq_len, d_model])
        output = decoder(input, context=context_emb)
        self.assertEqual(output.shape, (batch, seq_len, d_model))

    def test_auto_regressive_decode(self):
        batch_size, enc_seq_len, dff, cache_max_seq_len = 2, 4, 32, 4
        num_heads, head_size, vocab_size = 4, 8, 5
        d_model = num_heads * head_size
        decoder = Decoder(num_layers=4,
                          d_model=head_size,
                          num_heads=num_heads,
                          dff=dff,
                          vocab_size=vocab_size,
                          cache_max_seq_len=cache_max_seq_len)
        logits = tf.keras.layers.Dense(vocab_size)
        context = tf.random.normal([batch_size, enc_seq_len, d_model])
        decoder_input = tf.constant(1, shape=[batch_size, 1], dtype=tf.int32)
        output_array = tf.TensorArray(dtype=tf.int64,
                                      size=0,
                                      dynamic_size=True)
        for i in range(cache_max_seq_len):
            if i > 0:
                context = tf.random.normal([batch_size, 0, d_model])
            predictions = logits(
                decoder(decoder_input,
                        context,
                        reset_cache=tf.constant(True if i == 0 else False,
                                                shape=[batch_size, 1])))
            # get the maximum of axis -1
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i, predicted_id)
            decoder_input = predicted_id
        outputs = tf.transpose(tf.squeeze(output_array.stack(), axis=-1))
        self.assertEqual(outputs.shape, [batch_size, cache_max_seq_len])


if __name__ == '__main__':
    unittest.main()
