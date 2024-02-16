import tensorflow as tf

from keras.utils import register_keras_serializable
from typing import List


@register_keras_serializable(package="codeless_ml.ml.transformer")
class KVCache(tf.keras.layers.Layer):

    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int,
                 head_size: int):
        super(KVCache, self).__init__()
        (self._batch_size, self._max_seq_len, self._num_heads,
         self._head_size) = (batch_size, max_seq_len, num_heads, head_size)
        self._cache = tf.Variable(tf.zeros(
            [batch_size, max_seq_len, num_heads, head_size], dtype=tf.float32),
                                  trainable=False)
        self._index = tf.Variable(0, trainable=False)
        self._mask = tf.Variable(tf.constant(True,
                                             shape=[batch_size, max_seq_len]),
                                 trainable=False)

    @property
    def content(self):
        return self._cache[:, :self._index.value(), :, :]

    @property
    def mask(self):
        return self._mask[:, :self._index.value()]

    def _update_cache_and_mask(self,
                               x: tf.Tensor,
                               mask: tf.Tensor | None = None):
        from_idx = self._index.value()
        to_idx = from_idx + x.shape[1]
        # ensure we have enough space to hold the new key and values.
        tf.debugging.assert_less(to_idx, self._max_seq_len)
        left_size, right_size = from_idx, self._max_seq_len - to_idx
        padded_x = tf.pad(x, [[0, 0], [left_size, right_size], [0, 0], [0, 0]])
        # replace the cache with the given key and value tensor.
        self._cache.assign_add(padded_x)

        # update mask
        if mask is None:
            mask = tf.constant(True, shape=[self._batch_size, x.shape[1]])
        padded_mask = tf.pad(mask, [[0, 0], [left_size, right_size]],
                             constant_values=True)
        self._mask.assign(tf.math.logical_and(self._mask.value(), padded_mask))
        self._index.assign(to_idx)
        return self.content

    def call(self,
             x: tf.Tensor,
             reset_index: tf.Tensor,
             mask: tf.Tensor | None = None):
        tf.ensure_shape(
            x, [self._batch_size, None, self._num_heads, self._head_size])
        seq_len = x.shape[1]
        # update the index if we're requested to reset the index, which usually
        # happens for every new batch.
        self._index.assign(
            tf.cond(reset_index,
                    true_fn=lambda: 0,
                    false_fn=lambda: self._index.value()))
        self._cache.assign(
            tf.cond(reset_index,
                    true_fn=lambda: tf.zeros([
                        self._batch_size, self._max_seq_len, self._num_heads,
                        self._head_size
                    ]),
                    false_fn=lambda: self._cache))
        return self._update_cache_and_mask(x, mask)

    def compute_mask(self, x: tf.Tensor, mask=None):
        del x
        return self.mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "batch_size": self._batch_size,
            "max_seq_len": self._max_seq_len,
            "num_heads": self._num_heads,
            "head_size": self._head_size
        })
        return config
