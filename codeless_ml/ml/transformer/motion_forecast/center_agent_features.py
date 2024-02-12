import tensorflow as tf

import codeless_ml.ml.transformer.motion_forecast.center_agent_util as util
import codeless_ml.ml.transformer.motion_forecast.agent_features as af


def find_k_closest_entities(entity_xys: tf.Tensor, k=5):
    tf.ensure_shape(entity_xys, [None, None, 2])
    num_ego_agents, num_entities, d = entity_xys.shape
    # shape: [N, N ,2] where the first dimension is the ego agent and the second
    # dimension is other agents.
    # find up-to-k closest agents for each ego agent.
    # get the ego agents.
    ego_xys = tf.zeros([num_ego_agents, num_entities, 2])
    # shape: (N, N)
    # note: negate the distances as top_k returns k largest value while we want
    # k smallest distances.
    dists = -tf.math.reduce_euclidean_norm(ego_xys - entity_xys, axis=2)
    tf.ensure_shape(dists, [num_ego_agents, num_entities])
    # shape: (N, k)
    closest_vals, closest_indices = tf.math.top_k(dists, k=k)
    return tf.abs(closest_vals), closest_indices


def center_agent_features(feature_bundle: af.FeatureBundle):
    agent_cur_xys = feature_bundle.agent_cur_xys
    agent_cur_yaws = feature_bundle.agent_cur_yaws
    tf.ensure_shape(agent_cur_xys, [None, 1, 2])
    tf.ensure_shape(agent_cur_yaws, [None, 1, 1])
    # agent_rotations: [M, 2, 2]
    # agent_translations: [M, 2]
    num_agents = agent_cur_xys.shape[0]
    agent_rotations, agent_translations = util.agent_centric_transformation(
        tf.squeeze(agent_cur_xys, axis=1), tf.squeeze(agent_cur_yaws, axis=1))
    tf.ensure_shape(agent_rotations, [num_agents, 2, 2])
    tf.ensure_shape(agent_translations, [num_agents, 2])

    # the 3rd dim is time dimension which is 1 here.
    # new_agent_cur_xys: [M, M, 1, 2]
    # new_agent_cur_yaw_vecs: [M, M, 1, 2]
    new_agent_cur_xys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, agent_cur_xys)
    tf.ensure_shape(new_agent_cur_xys, [num_agents, num_agents, 1, 2])
    new_agent_cur_vxys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, feature_bundle.agent_cur_vxys)
    tf.ensure_shape(new_agent_cur_xys, [num_agents, num_agents, 1, 2])
    new_agent_cur_yaw_vecs = util.transform_multi_timestep_dirs(
        agent_rotations,
        util.transform_multi_timestep_yaws_to_vecs(agent_cur_yaws))
    tf.ensure_shape(new_agent_cur_yaw_vecs, [num_agents, num_agents, 1, 2])

    # centerize agent past positions, velocities and bbox yaws.
    agent_past_xys, agent_past_vxys, agent_past_yaws = (
        feature_bundle.agent_past_xys, feature_bundle.agent_past_vxys,
        feature_bundle.agent_past_yaws)
    new_agent_past_xys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, feature_bundle.agent_past_xys)
    new_agent_past_yaw_vecs = util.transform_multi_timestep_dirs(
        agent_rotations,
        util.transform_multi_timestep_yaws_to_vecs(agent_past_yaws))
    new_agent_past_vxys = util.transform_multi_timestep_dirs(
        agent_rotations, agent_past_vxys)
    # shape: [M, L, T, 2]
    tf.ensure_shape(
        new_agent_past_xys,
        [num_agents, agent_past_xys.shape[0], agent_past_xys.shape[1], 2])
    # shape: [M, L, T, 2]
    tf.ensure_shape(
        new_agent_past_yaw_vecs,
        [num_agents, agent_past_yaws.shape[0], agent_past_yaws.shape[1], 2])
    tf.ensure_shape(
        new_agent_past_vxys,
        [num_agents, agent_past_vxys.shape[0], agent_past_vxys.shape[1], 2])

    # centerize agent future positions, velocities and bbox yaws.
    agent_future_xys, agent_future_vxys, agent_future_yaws = (
        feature_bundle.agent_future_xys, feature_bundle.agent_future_vxys,
        feature_bundle.agent_future_yaws)
    new_agent_future_xys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, feature_bundle.agent_future_xys)
    new_agent_future_yaw_vecs = util.transform_multi_timestep_dirs(
        agent_rotations,
        util.transform_multi_timestep_yaws_to_vecs(agent_future_yaws))
    new_agent_future_vxys = util.transform_multi_timestep_dirs(
        agent_rotations, agent_future_vxys)
    # shape: [M, L, T, 2]
    tf.ensure_shape(
        new_agent_future_xys,
        [num_agents, agent_future_xys.shape[0], agent_future_xys.shape[1], 2])
    tf.ensure_shape(new_agent_future_yaw_vecs, [
        num_agents, agent_future_yaws.shape[0], agent_future_yaws.shape[1], 2
    ])
    tf.ensure_shape(new_agent_future_vxys, [
        num_agents, agent_future_vxys.shape[0], agent_future_vxys.shape[1], 2
    ])

    # centering the traffic light positions to ego agents.
    new_tl_cur_xys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, feature_bundle.tl_cur_xys)
    new_tl_past_xys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, feature_bundle.tl_past_xys)

    # centering the roadgraph points.
    rg_xys, rg_dirs = feature_bundle.rg_xys, feature_bundle.rg_dirs
    new_rg_xys = util.transform_multi_timestep_positions(
        agent_rotations, agent_translations, feature_bundle.rg_xys)
    new_rg_dirs = util.transform_multi_timestep_dirs(agent_rotations,
                                                     feature_bundle.rg_dirs)
    tf.ensure_shape(new_rg_xys, [num_agents, rg_xys.shape[0], 1, 2])
    tf.ensure_shape(new_rg_dirs, [num_agents, rg_dirs.shape[0], 1, 2])

    return feature_bundle._replace(
        agent_cur_yaw_vecs=new_agent_cur_yaw_vecs,
        agent_cur_xys=new_agent_cur_xys,
        agent_cur_vxys=new_agent_cur_vxys,
        agent_past_yaw_vecs=new_agent_past_yaw_vecs,
        agent_past_xys=new_agent_past_xys,
        agent_past_vxys=new_agent_past_vxys,
        agent_future_yaw_vecs=new_agent_future_yaw_vecs,
        agent_future_xys=new_agent_future_xys,
        agent_future_vxys=new_agent_future_vxys,
        tl_cur_xys=new_tl_cur_xys,
        tl_past_xys=new_tl_past_xys,
        rg_dirs=new_rg_dirs,
        rg_xys=new_rg_xys,
    )
