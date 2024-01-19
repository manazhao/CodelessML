import os
import unittest

import numpy as np
import tensorflow as tf

import codeless_ml.common.global_variable as gv
import codeless_ml.ml.transformer.loss_and_metric as loss_and_metric
import codeless_ml.common.callable_pb2 as callable_pb2

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


class InputTest(unittest.TestCase):

    def testLossFunction(self):
        callable_registry = callable_pb2.CallableRegistry()
        callable_registry.function_name = loss_and_metric.LOSS_REGESTRY_KEY
        loss_function = GVR.retrieve_callable(callable_registry)
        label = tf.constant([[1, 1, 1, 0, 0], [2, 2, 0, 0, 0]], dtype=tf.int32)
        pred = tf.random.uniform([2, 5, 3], dtype=tf.float32)
        loss = loss_function(label, pred)
        # loss should be a scalar.
        np.testing.assert_array_equal(tf.rank(loss), 0)
        np.testing.assert_array_equal(tf.size(loss), 1)

    def testAccuracyFunction(self):
        callable_registry = callable_pb2.CallableRegistry()
        callable_registry.function_name = loss_and_metric.METRIC_REGISTRY_KEY
        metric_function = GVR.retrieve_callable(callable_registry)
        label = tf.constant([[1, 1, 1, 0, 0], [2, 2, 0, 0, 0]], dtype=tf.int64)
        pred = tf.constant([[[0.1, 0.5, 0.4], [0.1, 0.5, 0.4], [0.1, 0.5, 0.4],
                             [1.0, 0, 0], [1.0, 0, 0]],
                            [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [1, 0, 0],
                             [1, 0, 0], [1, 0, 0]]],
                           dtype=tf.float32)
        accuracy = metric_function(label, pred)
        self.assertEqual(accuracy, 1.0)


if __name__ == '__main__':
    unittest.main()
