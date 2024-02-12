import os
import unittest

import math
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from typing import Tuple

import codeless_ml.ml.transformer.motion_forecast.center_agent_util as util

PI = math.pi


class CenterAgentUtilTest(unittest.TestCase):

    def setUp(self):
        self.xys = tf.constant([[1.0, .0], [.0, 1.0]])
        self.yaws = tf.constant([[PI / 2], [.0]])

    def test_transform_parameters(self):
        rotation, translation = util.agent_centric_transformation(
            self.xys, self.yaws)
        # The first dimension corresonds to agents.
        # the rotation matrix is 2x2 for a single agent and since we have two
        # agents, the shape is [2, 2, 2].
        self.assertEqual(rotation.shape, [2, 2, 2])
        self.assertEqual(translation.shape, [2, 2])
        npt.assert_almost_equal(
            rotation, [[[.0, 1.0], [-1.0, .0]], [[1., .0], [.0, 1.0]]])
        npt.assert_almost_equal(translation, [[.0, 1.0], [.0, -1.0]])

    def rotate_and_translate(self, rotation, translation, x):
        translation = tf.reshape(translation, [-1, 1])
        x = tf.reshape(x, [-1, 1])
        return tf.matmul(rotation, x) + translation

    def test_transform(self):
        rotation, translation = util.agent_centric_transformation(
            self.xys, self.yaws)
        new_xys = util.transform_position(rotation, translation, self.xys)
        new_yaw_vecs = util.transform_direction(rotation,
                                                util.yaws_to_vecs(self.yaws))
        # The first and second dimension correspond to the ego agent and the
        # target agent and the last dimension is the measurement, i.e., position
        # or yaw in unit vector.
        self.assertEqual(new_xys.shape, [2, 2, 2])
        self.assertEqual(new_yaw_vecs.shape, [2, 2, 2])

        xy2_in_1 = self.rotate_and_translate(rotation[0, :, :],
                                             translation[0, :], self.xys[1, :])
        xy1_in_2 = self.rotate_and_translate(rotation[1, :, :],
                                             translation[1, :], self.xys[0, :])
        xys_in_1 = tf.concat(
            [tf.zeros([1, 2]), tf.transpose(xy2_in_1)], axis=0)
        xys_in_2 = tf.concat(
            [tf.transpose(xy1_in_2), tf.zeros([1, 2])], axis=0)
        expected = tf.stack([xys_in_1, xys_in_2])
        npt.assert_allclose(new_xys, expected)


if __name__ == '__main__':
    unittest.main()
