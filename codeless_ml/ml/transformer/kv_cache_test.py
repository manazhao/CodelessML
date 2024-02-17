import numpy as np
import numpy.testing as npt
import tensorflow as tf

from codeless_ml.ml.transformer.kv_cache import KVCache


class KVCacheTest(tf.test.TestCase):

    def test_update_cache(self):
        batch_size, num_heads, head_size = 10, 2, 8
        layer = KVCache(max_seq_len=128,
                        num_heads=num_heads,
                        head_size=head_size)
        # now add values to cache.
        seq_len = 5
        x = np.random.rand(batch_size, seq_len, num_heads, head_size)
        content = layer(x, reset_index=tf.constant(False), mask=None)
        x_mask = content._keras_mask
        npt.assert_allclose(content, x)
        self.assertEqual(content.shape,
                         [batch_size, seq_len, num_heads, head_size])
        self.assertEqual(content._keras_mask.shape, [batch_size, seq_len])
        npt.assert_array_equal(content._keras_mask, content._keras_mask)
        self.assertTrue(np.all(content._keras_mask))

        # add another input to the cache with a different mask.
        y_len = 1
        y = np.random.rand(batch_size, y_len, num_heads, head_size)
        y_mask = np.random.choice([False, True],
                                  size=[batch_size, y_len],
                                  p=[0.5, 0.5])
        content = layer(y, reset_index=tf.constant(False), mask=y_mask)
        npt.assert_allclose(content, np.concatenate((x, y), axis=1))
        npt.assert_array_equal(content._keras_mask,
                               np.concatenate((x_mask, y_mask), axis=1))
        self.assertEqual(content._keras_mask.shape,
                         [batch_size, y_len + seq_len])

        # now reset the cache.
        content = layer(y, reset_index=tf.constant(True), mask=y_mask)
        npt.assert_allclose(content, y)
        npt.assert_array_equal(content._keras_mask, y_mask)


if __name__ == "__main__":
    tf.test.main()
