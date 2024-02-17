import math
import tensorflow as tf

from collections import namedtuple
from codeless_ml.ml.transformer.kv_cache import KVCache


# `CacheAttention` can cache the key and value matrices for MultiHeadAttention
# for existing tokens. Cache should only be used in model inference and can
# avoid unncessary computation and hence speed up inference.
# The implementation is based on
# https://github.com/tensorflow/models/blob/v2.15.0/official/nlp/modeling/layers/attention.py
# with modifications on the cache implementation. In particular, the reference
# requires the caller to maintain the cache externally, while the implementation
# here matains internally.
class CacheAttention(tf.keras.layers.MultiHeadAttention):

    def __init__(self,
                 num_heads: int,
                 key_dim: int,
                 value_dim: int | None = None,
                 cache_max_seq_len: int | None = None,
                 *args,
                 **kwargs):
        super().__init__(num_heads=num_heads,
                         key_dim=key_dim,
                         value_dim=value_dim,
                         *args,
                         **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        if value_dim is None:
            value_dim = key_dim
        self.value_dim = value_dim
        self._cache_max_seq_len = cache_max_seq_len
        self._key_cache, self._value_cache = (None, None)

    def _init_cache(self):
        if not self._cache_max_seq_len:
            return

        self._key_cache = KVCache(max_seq_len=self._cache_max_seq_len,
                                  num_heads=self.num_heads,
                                  head_size=self.key_dim)
        self._value_cache = KVCache(max_seq_len=self._cache_max_seq_len,
                                    num_heads=self.num_heads,
                                    head_size=self.key_dim)

    def _update_key_value_cache(self,
                                reset_cache: tf.Tensor,
                                key: tf.Tensor,
                                value: tf.Tensor,
                                key_mask: tf.Tensor | None = None,
                                value_mask: tf.Tensor | None = None):
        assert self._key_cache is not None and self._value_cache is not None, "cache objects must be created first."
        return (self._key_cache(key, reset_cache, mask=key_mask),
                self._value_cache(value, reset_cache, mask=value_mask))

    @property
    def key(self):
        if self._key_cache is None:
            return None

        return self._key_cache.content

    @property
    def value(self):
        if self._value_cache is None:
            return None

        return self._value_cache.content

    def call(self,
             query,
             value,
             key=None,
             attention_mask=None,
             return_attention_scores=False,
             training=None,
             use_causal_mask=False,
             reset_cache=None):
        # Note: the following code is copied from
        # https://github.com/tensorflow/models/blob/v2.15.0/official/nlp/modeling/layers/attention.py
        # with minor modifications.
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
            self._init_cache()

        if key is None:
            key = value

        # Convert RaggedTensor to Tensor.
        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()
        key_is_ragged, value_is_ragged = (
            isinstance(key, tf.RaggedTensor),
            isinstance(value, tf.RaggedTensor),
        )
        if key_is_ragged and value_is_ragged:
            # Ensure they have the same shape.
            bounding_shape = tf.math.maximum(key.bounding_shape(),
                                             value.bounding_shape())
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))

        use_kv_cache = self._cache_max_seq_len is not None
        attention_mask = None
        if not use_kv_cache:
            attention_mask = self._compute_attention_mask(
                query,
                value,
                key=key,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, F, N ,H]
        query_mask = (query._keras_mask
                      if hasattr(query, '_keras_mask') else None)
        key_mask = (key._keras_mask if hasattr(key, '_keras_mask') else None)
        value_mask = (value._keras_mask
                      if hasattr(value, '_keras_mask') else None)
        query = self._query_dense(query)
        setattr(query, '_keras_mask', query_mask)
        # `key` = [B, T, N, H]
        key = self._key_dense(key)
        # `value` = [B, T, N, H]
        value = self._value_dense(value)
        if use_kv_cache:
            # get the full-length key and value tensors by joining cached key and
            # value with the new keys and values.
            # Note: the masks for key and value can be accessed via their
            # _keras_mask attribute.
            key, value = self._update_key_value_cache(reset_cache[0, 0],
                                                      key,
                                                      value,
                                                      key_mask=key_mask,
                                                      value_mask=value_mask)
            # we want to compute the mask with the full-length key and value
            # tensors.
            attention_mask = self._compute_attention_mask(
                query,
                value,
                key=key,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
            )
        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training)
        attention_output = self._output_dense(attention_output)
        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output
