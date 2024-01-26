import unittest

import numpy as np
import tensorflow as tf

from codeless_ml.ml.registry.mnist import return_prepare_image_input_fn


class RegisterCallableTest(unittest.TestCase):

    def setUp(self):
        super(RegisterCallableTest, self).setUp()
        self._feature_value = np.array([[1., 2., 3., 4.]])
        self._label_value = np.array([[0]])
        self._feature_map = {
            "x": tf.constant(self._feature_value),
            "y": tf.constant(self._label_value),
            "z": tf.constant([.1, 0.2]),
            "width": tf.constant([[2]]),
            "height": tf.constant([[2]]),
            "channel": tf.constant([[1]])
        }

    def testSlice(self):
        map_fn = return_prepare_image_input_fn(feature_name="x",
                                               label_name="y",
                                               width_name="width",
                                               height_name="height",
                                               channel_name="channel")
        feature, label = map_fn(self._feature_map)
        np.testing.assert_array_equal(
            np.reshape(self._feature_value, (1, 2, 2, 1)), feature.numpy())
        np.testing.assert_array_equal(self._label_value, label.numpy())


if __name__ == '__main__':
    unittest.main()
