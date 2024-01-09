import os
import unittest

import numpy as np
import tensorflow as tf

import codeless_ml.ml.transformer.model as model


class InputTest(unittest.TestCase):

  def testMasks_BadInputShape(self):
    seq = tf.constant(np.arange(4), dtype=tf.float32)
    with self.assertRaises(ValueError):
      model.create_padding_mask(seq)

  def testMasks_GoodInputShape(self):
    seq = tf.constant([[1, 2, 3, 0], [3, 0, 0, 1]], dtype=tf.float32)
    mask = model.create_padding_mask(seq)
    np.testing.assert_array_equal(tf.shape(mask), [2, 1, 4])
    np.testing.assert_array_equal(np.matrix.flatten(mask.numpy()),
                                  [0, 0, 0, 1, 0, 1, 1, 0])

  def testLookAheadMask(self):
    mask = model.create_look_ahead_mask(3)
    np.testing.assert_array_equal(mask, [[[0, 1, 1], [0, 0, 1], [0, 0, 0]]])

  def testScaledProductAttention(self):
    # Create two query vectors, each matching to the first and the second row of
    # the key vectors. The resulting attention weights are [1, 0, 0] and [0, 1,
    # 0] respectively.
    q = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
    k = tf.constant([[1e6, 0, 0], [0, 1e6, 0], [0, 0, 1e6]], dtype=tf.float32)
    v = tf.constant([[1], [2], [3]], dtype=tf.float32)
    output, attention_weights = model.scaled_dot_product_attention(q=q,
                                                                   k=k,
                                                                   v=v,
                                                                   mask=None)
    np.testing.assert_almost_equal(attention_weights, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(output.numpy(), [[1], [2]])

  def testScaledProductAttention_Mask(self):
    q = tf.constant([[1, 0, 0]], dtype=tf.float32)
    k = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    v = tf.constant([[1], [2], [3]], dtype=tf.float32)
    mask = model.create_padding_mask(tf.constant([[1, 0, 0]], dtype=tf.float32))
    output, attention_weights = model.scaled_dot_product_attention(q=q,
                                                                   k=k,
                                                                   v=v,
                                                                   mask=mask)
    # output shape: [batch_size, seq_len, d_model]
    # or [1, 1, 3]
    np.testing.assert_almost_equal(attention_weights, [[[1, 0, 0]]])
    np.testing.assert_almost_equal(output, [[[1]]])

  def testMultiHeadAttentionLayer(self):
    d_model = 10
    num_heads = 5
    depth = d_model // num_heads
    mha = model.MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    # q shape: [batch_size, seq_len, embedding_dim] = [5, 2, 2]
    batch_size = 5
    seq_len = 2
    embedding_dim = 2
    q = tf.reshape(tf.constant(np.arange(batch_size * seq_len * embedding_dim),
                               dtype=tf.float32),
                   shape=(batch_size, seq_len, embedding_dim))
    output, attention_weights = mha(q=q, k=q, v=q, mask=None)
    np.testing.assert_array_equal(tf.shape(output),
                                  [batch_size, seq_len, d_model])
    np.testing.assert_array_equal(tf.shape(attention_weights),
                                  [batch_size, num_heads, seq_len, depth])

  def testFeedForwardNetwork(self):
    d_model = 10
    dff = 5
    ffn = model.point_wise_feed_forward_network(dff=dff, d_model=d_model)
    batch_size = 32
    seq_len = 10
    embedding_dim = 2
    output = ffn(tf.random.uniform((batch_size, seq_len, embedding_dim)))
    np.testing.assert_array_equal(tf.shape(output),
                                  [batch_size, seq_len, d_model])

  def testEncoderLayer(self):
    d_model = 10
    num_heads = 5
    dff = 30
    batch_size = 32
    seq_len = 10
    embedding_dim = 10
    encoder = model.EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff)
    x = tf.random.uniform((batch_size, seq_len, embedding_dim))
    output = encoder(x, training=False, mask=None)
    np.testing.assert_array_equal(tf.shape(output),
                                  [batch_size, seq_len, d_model])

  def testEncoder(self):
    d_model = 10
    num_heads = 5
    batch_size = 32
    seq_len = 10
    embedding_dim = 10
    dff = 30

    x = tf.random.uniform((batch_size, seq_len))
    encoder = model.Encoder(num_layers=2,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            vocab_size=5000,
                            max_position_encoding=10000)
    output = encoder(x, training=False, mask=None)
    np.testing.assert_array_equal(tf.shape(output),
                                  [batch_size, seq_len, d_model])

  def testDecoderLayer(self):
    d_model = 10
    num_heads = 5
    batch_size = 32
    seq_len = 10
    embedding_dim = 10
    dff = 30

    x = tf.random.uniform((batch_size, seq_len, embedding_dim))
    encoder = model.EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff)
    enc_output = encoder(x, training=False, mask=None)
    decoder = model.DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff)
    output, _, _ = decoder(x=x,
                           enc_output=enc_output,
                           training=False,
                           look_ahead_mask=None,
                           padding_mask=None)
    np.testing.assert_array_equal(tf.shape(output),
                                  [batch_size, seq_len, d_model])

  def testDecoder(self):
    d_model = 10
    num_heads = 5
    vocab_size = 500
    batch_size = 32
    seq_len = 10
    embedding_dim = 10
    dff = 30
    decoder = model.Decoder(num_layers=2,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            vocab_size=vocab_size,
                            max_position_encoding=500000)
    target_seq_len = 20
    decoder_input = tf.random.uniform((batch_size, target_seq_len),
                                      dtype=tf.int64,
                                      minval=0,
                                      maxval=200)
    enc_output = tf.random.uniform((batch_size, seq_len, d_model))
    output, attn = decoder(x=decoder_input,
                           enc_output=enc_output,
                           training=False,
                           look_ahead_mask=None,
                           padding_mask=None)
    np.testing.assert_array_equal(tf.shape(output),
                                  [batch_size, target_seq_len, d_model])

  def testTransformer(self):
    num_layers = 2
    d_model = 10
    num_heads = 5
    embedding_dim = 10
    dff = 30

    input_vocab_size = 500
    target_vocab_size = 1000
    max_pe_input = 1000
    max_pe_target = 2000

    sample_model = model.Transformer(num_layers=num_layers,
                                     d_model=d_model,
                                     num_heads=num_heads,
                                     dff=dff,
                                     input_vocab_size=input_vocab_size,
                                     target_vocab_size=target_vocab_size,
                                     max_pe_input=max_pe_input,
                                     max_pe_target=max_pe_target)

    batch_size = 32
    seq_len = 10
    target_seq_len = 20

    temp_input = tf.random.uniform((batch_size, seq_len),
                                   dtype=tf.int64,
                                   minval=0,
                                   maxval=200)
    temp_target = tf.random.uniform((batch_size, target_seq_len),
                                    dtype=tf.int64,
                                    minval=0,
                                    maxval=200)

    fn_out, _ = sample_model([temp_input, temp_target], training=False)
    np.testing.assert_array_equal(
        tf.shape(fn_out), [batch_size, target_seq_len, target_vocab_size])


if __name__ == '__main__':
  unittest.main()
