import numpy as np
import numpy.testing as npt
import tensorflow as tf

import codeless_ml.ml.transformer.cache_attention as ca


class CacheAttentionTest(tf.test.TestCase):

    def test_no_cache(self):
        num_heads, key_dim = 2, 8
        layer = ca.CacheAttention(num_heads=num_heads, key_dim=key_dim)

        batch_size, seq_len = 4, 10
        d_model = key_dim * num_heads

        query = np.random.rand(batch_size, seq_len, d_model)
        attn, attn_scores = layer(query=query,
                                  value=query,
                                  return_attention_scores=True)
        self.assertEqual(attn.shape, [batch_size, seq_len, d_model])
        self.assertEqual(attn_scores.shape,
                         [batch_size, num_heads, seq_len, seq_len])

    def test_with_cache(self):
        batch_size, num_heads, key_dim = 4, 2, 8
        max_seq_len = 128
        layer = ca.CacheAttention(num_heads=num_heads,
                                  key_dim=key_dim,
                                  cache_max_seq_len=max_seq_len)
        # cache gets only initialized when the layer gets called for the first
        # time.
        self.assertTrue(layer.key is None)
        self.assertTrue(layer.value is None)

        batch_size, seq_len = 4, 10
        d_model = key_dim * num_heads

        query = np.random.rand(batch_size, seq_len, d_model)
        # don't reset cache.
        attn, attn_scores = layer(query=query,
                                  value=query,
                                  reset_cache=tf.zeros([batch_size, 1],
                                                       dtype=tf.int32),
                                  return_attention_scores=True)
        self.assertEqual(layer.key.shape,
                         [batch_size, seq_len, num_heads, key_dim])
        self.assertEqual(layer.value.shape,
                         [batch_size, seq_len, num_heads, key_dim])

        self.assertEqual(attn.shape, [batch_size, seq_len, d_model])
        self.assertEqual(attn_scores.shape,
                         [batch_size, num_heads, seq_len, seq_len])

        single_token_query = np.random.rand(batch_size, 1, d_model)
        new_attn, new_attn_scores = layer(single_token_query,
                                          value=single_token_query,
                                          reset_cache=tf.zeros([batch_size, 1],
                                                               dtype=tf.int32),
                                          return_attention_scores=True)
        self.assertEqual(new_attn.shape, [batch_size, 1, d_model])
        self.assertEqual(new_attn_scores.shape,
                         [batch_size, num_heads, 1, seq_len + 1])
        self.assertEqual(layer.key.shape,
                         [batch_size, seq_len + 1, num_heads, key_dim])
        self.assertEqual(layer.value.shape,
                         [batch_size, seq_len + 1, num_heads, key_dim])

    def test_with_mask(self):
        num_heads, key_dim = 2, 8
        query = tf.constant([[1, 0]], dtype=tf.int32)
        d_model = num_heads * key_dim
        emb_layer = tf.keras.layers.Embedding(input_dim=10,
                                              output_dim=d_model,
                                              mask_zero=True)
        query_emb = emb_layer(query)
        value = tf.constant([[1, 1, 1, 0]], dtype=tf.int32)
        value_emb = emb_layer(value)

        batch_size, max_seq_len = query.shape[0], 128
        attention_layer = ca.CacheAttention(num_heads=num_heads,
                                            key_dim=key_dim,
                                            cache_max_seq_len=max_seq_len)
        reset_cache = tf.zeros([batch_size, 1], dtype=tf.int32)
        attn, attn_scores = attention_layer(query_emb,
                                            value=value_emb,
                                            reset_cache=reset_cache,
                                            return_attention_scores=True)
        non_padding_token_scores = tf.constant([1.0, 1.0, 1.0, .0])
        one_head_scores = tf.stack([
            non_padding_token_scores / tf.reduce_sum(non_padding_token_scores),
            tf.ones([4]) / 4
        ])
        npt.assert_allclose(
            attn_scores,
            tf.expand_dims(tf.stack([one_head_scores, one_head_scores]),
                           axis=0))
        npt.assert_array_equal(attn._keras_mask, [[True, False]])

        # now add two more tokens and check the attention output.
        for token_id in range(2, 4):
            query_emb = emb_layer(tf.constant([[token_id]], dtype=tf.int32))
            attn, attn_scores = attention_layer(query_emb,
                                                value=query_emb,
                                                reset_cache=reset_cache,
                                                return_attention_scores=True)
            non_padding_token_scores = tf.concat(
                [non_padding_token_scores, [1.0]], axis=0)
            one_head_scores = tf.expand_dims(
                non_padding_token_scores /
                tf.reduce_sum(non_padding_token_scores),
                axis=0)
            npt.assert_allclose(attn_scores,
                                tf.expand_dims(tf.stack(
                                    [one_head_scores, one_head_scores]),
                                               axis=0),
                                atol=0.001)


if __name__ == "__main__":
    tf.test.main()
