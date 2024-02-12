import math
import unittest
import numpy.testing as npt
import tensorflow as tf

import codeless_ml.ml.transformer.motion_forecast.agent_features as af
import codeless_ml.ml.transformer.motion_forecast.center_agent_util as util
import codeless_ml.ml.transformer.motion_forecast.center_agent_features as caf


class AgentFeaturesTest(unittest.TestCase):

    def setUp(self):
        cur_state = lambda name: f"state/current/{name}"
        past_state = lambda name: f"state/past/{name}"
        future_state = lambda name: f"state/future/{name}"
        # traffic light state
        tl_cur_state = lambda name: f"traffic_light_state/current/{name}"
        tl_past_state = lambda name: f"traffic_light_state/past/{name}"
        rg_state = lambda name: f"roadgraph_samples/{name}"
        self.inputs = {
            'state/type':
            tf.constant([1, 2, -1]),
            'state/is_sdc':
            tf.constant([0, 1, -1]),
            # current state features.
            cur_state('length'):
            tf.constant([[1.0], [2.0], [-1]]),
            cur_state('width'):
            tf.constant([[3.0], [4.0], [-1]]),
            cur_state('bbox_yaw'):
            tf.constant([[0.1], [0.2], [-1]]),
            cur_state('x'):
            tf.constant([[5.0], [6.0], [-1]]),
            cur_state('y'):
            tf.constant([[7.0], [8.0], [-1]]),
            cur_state('velocity_x'):
            tf.constant([[9.0], [10.0], [-1]]),
            cur_state('velocity_y'):
            tf.constant([[11.0], [12.0], [-1]]),
            cur_state('valid'):
            tf.constant([[1], [1], [-1]]),
            # past state features.
            past_state('length'):
            tf.constant([[1.0, 1.0], [2.0, 2.0], [-1, -1]]),
            past_state('width'):
            tf.constant([[3.0, 3.0], [4.0, 4.0], [-1, -1]]),
            past_state('bbox_yaw'):
            tf.constant([[0.1, 0.1], [0.2, 0.2], [-1, -1]]),
            past_state('x'):
            tf.constant([[5.0, 5.0], [6.0, 6.0], [-1, -1]]),
            past_state('y'):
            tf.constant([[7.0, 7.0], [8.0, 8.0], [-1, -1]]),
            past_state('velocity_x'):
            tf.constant([[9.0, 9.0], [10.0, 10.0], [-1, -1]]),
            past_state('velocity_y'):
            tf.constant([[11.0, 11.0], [12.0, 12.0], [-1, -1]]),
            past_state('valid'):
            tf.constant([[1, 1], [1, 1], [-1, -1]]),
            # future state features
            future_state('length'):
            tf.constant([[12.0, 12.0], [13.0, 13.0], [-1, -1]]),
            future_state('width'):
            tf.constant([[14.0, 14.0], [15.0, 15.0], [-1, -1]]),
            future_state('bbox_yaw'):
            tf.constant([[0.1, 0.1], [0.2, 0.2], [-1, -1]]),
            future_state('x'):
            tf.constant([[16.0, 16.0], [17.0, 17.0], [-1, -1]]),
            future_state('y'):
            tf.constant([[18.0, 18.0], [19.0, 19.0], [-1, -1]]),
            future_state('velocity_x'):
            tf.constant([[20.0, 20.0], [21.0, 21.0], [-1, -1]]),
            future_state('velocity_y'):
            tf.constant([[22.0, 22.0], [23.0, 23.0], [-1, -1]]),
            future_state('valid'):
            tf.constant([[1, 1], [1, 1], [-1, -1]]),
            # traffic light state.
            tl_cur_state('state'):
            tf.constant([[1, 1]]),
            tl_cur_state('valid'):
            tf.constant([[1, -1]]),
            tl_cur_state('x'):
            tf.constant([[1.0, -1.0]]),
            tl_cur_state('y'):
            tf.constant([[2.0, -1.0]]),
            # tl past state
            tl_past_state('state'):
            tf.constant([[2, -1], [3, -1]]),
            tl_past_state('valid'):
            tf.constant([[1, -1], [1, -1]]),
            tl_past_state('x'):
            tf.constant([[2.0, -1.0], [3.0, -1]]),
            tl_past_state('y'):
            tf.constant([[4.0, -1.0], [5.0, -1]]),
            # rg samples
            rg_state('dir'):
            tf.constant([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [-1.0, -1.0,
                                                            -1.0]]),
            rg_state('type'):
            tf.constant([[1], [2], [-1]]),
            rg_state('valid'):
            tf.constant([[1], [1], [-1]]),
            rg_state('xyz'):
            tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1., -1., -1.]]),
        }

    def test_k_closest_entities(self):
        PI = math.pi
        xys = tf.constant([[1., .0], [.0, 1.0], [2.0, .0]])
        yaws = tf.constant([[.0], [PI / 2], [.0]])
        rotation, translation = util.agent_centric_transformation(xys, yaws)
        new_xys = util.transform_position(rotation, translation, xys)
        closest_vals, closest_indices = caf.find_k_closest_entities(new_xys,
                                                                    k=2)
        npt.assert_allclose(closest_vals, [[0., 1.0], [0., 1.414], [0., 1.]],
                            atol=0.01)
        npt.assert_almost_equal(closest_indices, [[0, 2], [1, 0], [2, 0]])

    def test_remove_invalid_agents(self):
        feature_bundle = af.remove_invalid_agents_and_map_features(self.inputs)
        npt.assert_almost_equal(feature_bundle.type, [[1], [2]])
        npt.assert_almost_equal(feature_bundle.is_sdc, [[0], [1]])
        # current features
        npt.assert_almost_equal(feature_bundle.agent_cur_ls,
                                [[[1.0]], [[2.0]]])
        npt.assert_almost_equal(feature_bundle.agent_cur_ws,
                                [[[3.0]], [[4.0]]])
        npt.assert_almost_equal(feature_bundle.agent_cur_yaws,
                                [[[0.1]], [[0.2]]])
        npt.assert_almost_equal(feature_bundle.agent_cur_xys,
                                [[[5.0, 7.0]], [[6.0, 8.0]]])
        npt.assert_almost_equal(feature_bundle.agent_cur_vxys,
                                [[[9.0, 11.0]], [[10.0, 12.0]]])
        # past features
        npt.assert_allclose(
            feature_bundle.agent_past_ls,
            [
                [[1.0], [1.0]],  # agent 1
                [[2.0], [2.0]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_past_ws,
            [
                [[3.0], [3.0]],  # agent 1
                [[4.0], [4.0]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_past_yaws,
            [
                [[0.1], [0.1]],  # agent 1
                [[0.2], [0.2]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_past_xys,
            [
                [[5.0, 7.0], [5.0, 7.0]],  # agent 1
                [[6.0, 8.0], [6.0, 8.0]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_past_vxys,
            [
                [[9.0, 11.0], [9.0, 11.0]],  # agent 1
                [[10.0, 12.0], [10.0, 12.0]]  # agent 2
            ])
        npt.assert_almost_equal(feature_bundle.agent_past_valid,
                                [[1, 1], [1, 1]])
        # future features
        npt.assert_almost_equal(
            feature_bundle.agent_future_ls,
            [
                [[12.0], [12.0]],  # agent 1
                [[13.0], [13.0]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_future_ws,
            [
                [[14.0], [14.0]],  # agent 1
                [[15.0], [15.0]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_future_yaws,
            [
                [[0.1], [0.1]],  # agent 1
                [[0.2], [0.2]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_future_xys,
            [
                [[16.0, 18.0], [16.0, 18.0]],  # agent 1
                [[17.0, 19.0], [17.0, 19.0]]  # agent 2
            ])
        npt.assert_almost_equal(
            feature_bundle.agent_future_vxys,
            [
                [[20.0, 22.0], [20.0, 22.0]],  # agent 1
                [[21.0, 23.0], [21.0, 23.0]]  # agent 2
            ])
        npt.assert_almost_equal(feature_bundle.agent_future_valid,
                                [[1, 1], [1, 1]])
        # traffic light tensors
        npt.assert_almost_equal(feature_bundle.tl_cur_states, [[[1]]])
        npt.assert_almost_equal(
            feature_bundle.tl_cur_xys,
            [
                [[1.0, 2.0]],  # tl 1
            ])
        npt.assert_almost_equal(
            feature_bundle.tl_past_states,
            [
                [[2], [3]],  # tl 1
            ])
        npt.assert_almost_equal(
            feature_bundle.tl_past_xys,
            [
                [
                    [2.0, 4.0],  # time 1
                    [3.0, 5.0]  # time 2
                ],  # tl 1
            ])

        # map features
        npt.assert_almost_equal(
            feature_bundle.rg_xys,
            [
                [[1.0, 2.0]],  # sample 1
                [[4.0, 5.0]],  # sample 2
            ])
        npt.assert_almost_equal(
            feature_bundle.rg_dirs,
            [
                [[0.1, 0.1]],  # sample 1
                [[0.2, 0.2]],  # sample 2
            ])
        npt.assert_almost_equal(
            feature_bundle.rg_types,
            [
                [[1]],  # sample 1
                [[2]],  # sample 2
            ])

    def test_center_agent_features(self):
        feature_bundle = af.remove_invalid_agents_and_map_features(self.inputs)
        updated_feature_bundle = caf.center_agent_features(feature_bundle)
        # agent cur features
        self.assertEqual(updated_feature_bundle.agent_cur_xys.shape,
                         [2, 2, 1, 2])
        self.assertEqual(updated_feature_bundle.agent_cur_vxys.shape,
                         [2, 2, 1, 2])
        self.assertEqual(updated_feature_bundle.agent_cur_yaw_vecs.shape,
                         [2, 2, 1, 2])
        # agent past features
        self.assertEqual(updated_feature_bundle.agent_past_xys.shape,
                         [2, 2, 2, 2])
        self.assertEqual(updated_feature_bundle.agent_past_vxys.shape,
                         [2, 2, 2, 2])
        self.assertEqual(updated_feature_bundle.agent_past_yaw_vecs.shape,
                         [2, 2, 2, 2])
        # agent future features
        self.assertEqual(updated_feature_bundle.agent_future_xys.shape,
                         [2, 2, 2, 2])
        self.assertEqual(updated_feature_bundle.agent_future_vxys.shape,
                         [2, 2, 2, 2])
        self.assertEqual(updated_feature_bundle.agent_future_yaw_vecs.shape,
                         [2, 2, 2, 2])

        # tl
        self.assertEqual(updated_feature_bundle.tl_cur_xys.shape, [2, 1, 1, 2])
        self.assertEqual(updated_feature_bundle.tl_past_xys.shape,
                         [2, 1, 2, 2])
        # rg samples
        self.assertEqual(updated_feature_bundle.rg_xys.shape, [2, 2, 1, 2])
        self.assertEqual(updated_feature_bundle.rg_dirs.shape, [2, 2, 1, 2])


if __name__ == '__main__':
    unittest.main()
