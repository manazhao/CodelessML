import tensorflow as tf

from collections import namedtuple
from typing import Dict

FeatureBundle = namedtuple(
    'FeatureBundle',
    [
        'type',
        'is_sdc',
        'agent_cur_xys',
        'agent_cur_vxys',
        'agent_cur_ws',
        'agent_cur_ls',
        'agent_cur_yaws',
        'agent_cur_yaw_vecs',  # only filled after centering agents.
        'agent_past_xys',
        'agent_past_vxys',
        'agent_past_ws',
        'agent_past_ls',
        'agent_past_yaws',
        'agent_past_yaw_vecs',  # only filled after centering agents.
        'agent_past_valid',
        'agent_future_xys',
        'agent_future_vxys',
        'agent_future_ws',
        'agent_future_ls',
        'agent_future_yaws',
        'agent_future_yaw_vecs',  # only filled after centering agents.
        'agent_future_valid',
        'tl_cur_xys',
        'tl_cur_states',
        'tl_cur_valid',
        'tl_past_xys',
        'tl_past_states',
        'tl_past_valid',
        'rg_xys',
        'rg_dirs',
        'rg_types',
    ])


def remove_invalid_agents_and_map_features(
        inputs: Dict[str, tf.Tensor]) -> FeatureBundle:

    def _mask_and_merge(xs: tf.Tensor, ys: tf.Tensor, mask: tf.Tensor):
        mask = tf.squeeze(mask > 0)
        tf.ensure_shape(xs, [None, None])
        tf.ensure_shape(ys, [None, None])
        dims = xs.shape
        # if the 2nd dimension has more than one value, these values are usually
        # from different timestamps. As a result, we want to extract the time
        # time dimension as the 2nd dimension.
        xs = tf.reshape(xs, [dims[0], -1, 1])
        ys = tf.reshape(ys, [dims[0], -1, 1])
        xs = tf.boolean_mask(xs, mask)
        ys = tf.boolean_mask(ys, mask)
        tf.ensure_shape(xs, ys.shape)
        # concatenate xs and ys in the column direction.
        return tf.concat([xs, ys], axis=2)

    def _expand_and_merge(xs: tf.Tensor, ys: tf.Tensor):
        dims = xs.shape[0]
        xs = tf.reshape(xs, [dims, -1, 1])
        ys = tf.reshape(ys, [dims, -1, 1])
        return tf.concat([xs, ys], axis=2)

    def _mask(x: tf.Tensor, mask: tf.Tensor):
        return tf.boolean_mask(x, mask > 0)

    # agent current state
    (
        agent_cur_xs,
        agent_cur_ys,
        agent_cur_vxs,
        agent_cur_vys,
        agent_cur_ws,
        agent_cur_ls,
        agent_cur_yaws,
        valid_agents,
    ) = tuple([
        inputs[f"state/current/{k}"] for k in [
            'x',
            'y',
            'velocity_x',
            'velocity_y',
            'width',
            'length',
            'bbox_yaw',
            'valid',
        ]
    ])
    valid_agents = tf.squeeze(valid_agents)
    agent_cur_xys = _mask_and_merge(agent_cur_xs,
                                    agent_cur_ys,
                                    mask=valid_agents)
    agent_cur_vxys = _mask_and_merge(agent_cur_vxs,
                                     agent_cur_vys,
                                     mask=valid_agents)
    agent_cur_ws = tf.expand_dims(_mask(agent_cur_ws, mask=valid_agents),
                                  axis=2)
    agent_cur_ls = tf.expand_dims(_mask(agent_cur_ls, mask=valid_agents),
                                  axis=2)
    agent_cur_yaws = tf.expand_dims(_mask(agent_cur_yaws, mask=valid_agents),
                                    axis=2)

    # agent past states
    (
        agent_past_xs,
        agent_past_ys,
        agent_past_vxs,
        agent_past_vys,
        agent_past_ws,
        agent_past_ls,
        agent_past_yaws,
        agent_past_valid,
    ) = tuple([
        inputs[f"state/past/{k}"] for k in [
            'x',
            'y',
            'velocity_x',
            'velocity_y',
            'width',
            'length',
            'bbox_yaw',
            'valid',
        ]
    ])
    agent_past_xys = _mask_and_merge(agent_past_xs,
                                     agent_past_ys,
                                     mask=valid_agents)
    agent_past_vxys = _mask_and_merge(agent_past_vxs,
                                      agent_past_vys,
                                      mask=valid_agents)
    agent_past_ws = tf.expand_dims(_mask(agent_past_ws, mask=valid_agents),
                                   axis=2)
    agent_past_ls = tf.expand_dims(_mask(agent_past_ls, mask=valid_agents),
                                   axis=2)
    agent_past_yaws = tf.expand_dims(_mask(agent_past_yaws, mask=valid_agents),
                                     axis=2)
    agent_past_valid = _mask(agent_past_valid, mask=valid_agents)

    # agent future states
    (
        agent_future_xs,
        agent_future_ys,
        agent_future_vxs,
        agent_future_vys,
        agent_future_ws,
        agent_future_ls,
        agent_future_yaws,
        agent_future_valid,
    ) = tuple([
        inputs[f"state/future/{k}"] for k in [
            'x',
            'y',
            'velocity_x',
            'velocity_y',
            'width',
            'length',
            'bbox_yaw',
            'valid',
        ]
    ])
    agent_future_xys = _mask_and_merge(agent_future_xs,
                                       agent_future_ys,
                                       mask=valid_agents)
    agent_future_vxys = _mask_and_merge(agent_future_vxs,
                                        agent_future_vys,
                                        mask=valid_agents)
    agent_future_ws = tf.expand_dims(_mask(agent_future_ws, mask=valid_agents),
                                     axis=2)
    agent_future_ls = tf.expand_dims(_mask(agent_future_ls, mask=valid_agents),
                                     axis=2)
    agent_future_yaws = tf.expand_dims(_mask(agent_future_yaws,
                                             mask=valid_agents),
                                       axis=2)
    agent_future_valid = _mask(agent_future_valid, mask=valid_agents)

    # traffic light current state
    tl_cur_xs, tl_cur_ys, tl_cur_states, tl_cur_valid = tuple([
        inputs[f"traffic_light_state/current/{k}"]
        for k in ['x', 'y', 'state', 'valid']
    ])
    # transpose the tensor to [# of tls, T]
    # shape: [N, 2]
    tl_cur_valid = tf.squeeze(tl_cur_valid)
    tl_cur_xys = _mask_and_merge(tf.transpose(tl_cur_xs),
                                 tf.transpose(tl_cur_ys),
                                 mask=tl_cur_valid)
    tl_cur_states = tf.expand_dims(_mask(tf.transpose(tl_cur_states),
                                         mask=tl_cur_valid),
                                   axis=2)
    # traffic light past state
    tl_past_xs, tl_past_ys, tl_past_states, tl_past_valid = tuple([
        inputs[f"traffic_light_state/past/{k}"]
        for k in ['x', 'y', 'state', 'valid']
    ])
    tl_past_xys = _mask_and_merge(tf.transpose(tl_past_xs),
                                  tf.transpose(tl_past_ys),
                                  mask=tl_cur_valid)
    tl_past_states = tf.expand_dims(tf.transpose(tl_past_states), axis=2)
    tl_past_states = _mask(tl_past_states, mask=tl_cur_valid)
    tf_past_valid = tf.expand_dims(tf.transpose(tl_past_valid), axis=2)
    tl_past_valid = _mask(tf.transpose(tl_past_valid), mask=tl_cur_valid)

    # map features
    rg_valid, rg_xyzs, rg_types, rg_dirs = tuple([
        inputs[f"roadgraph_samples/{name}"] for name in [
            'valid',
            'xyz',
            'type',
            'dir',
        ]
    ])
    rg_valid = tf.squeeze(rg_valid)
    rg_xys = _mask(tf.expand_dims(rg_xyzs[:, :2], axis=1), mask=rg_valid)
    rg_dirs = _mask(tf.expand_dims(rg_dirs[:, :2], axis=1), mask=rg_valid)
    rg_types = _mask(tf.expand_dims(rg_types, axis=1), mask=rg_valid)
    agent_type, is_sdc = inputs['state/type'], inputs['state/is_sdc']
    tf.ensure_shape(agent_type, [
        None,
    ])
    tf.ensure_shape(is_sdc, [
        None,
    ])
    agent_type = tf.reshape(
        tf.boolean_mask(agent_type, tf.reshape(valid_agents > 0, [-1])),
        [-1, 1])
    is_sdc = tf.reshape(
        tf.boolean_mask(is_sdc, tf.reshape(valid_agents > 0, [-1])), [-1, 1])

    return FeatureBundle(type=agent_type,
                         is_sdc=is_sdc,
                         agent_cur_yaw_vecs=None,
                         agent_past_yaw_vecs=None,
                         agent_future_yaw_vecs=None,
                         agent_cur_xys=agent_cur_xys,
                         agent_cur_vxys=agent_cur_vxys,
                         agent_cur_ws=agent_cur_ws,
                         agent_cur_ls=agent_cur_ls,
                         agent_cur_yaws=agent_cur_yaws,
                         agent_past_xys=agent_past_xys,
                         agent_past_vxys=agent_past_vxys,
                         agent_past_ws=agent_past_ws,
                         agent_past_ls=agent_past_ls,
                         agent_past_yaws=agent_past_yaws,
                         agent_past_valid=agent_past_valid,
                         agent_future_xys=agent_future_xys,
                         agent_future_vxys=agent_future_vxys,
                         agent_future_ws=agent_future_ws,
                         agent_future_ls=agent_future_ls,
                         agent_future_yaws=agent_future_yaws,
                         agent_future_valid=agent_future_valid,
                         tl_cur_xys=tl_cur_xys,
                         tl_cur_states=tl_cur_states,
                         tl_cur_valid=tl_cur_valid,
                         tl_past_xys=tl_past_xys,
                         tl_past_states=tl_past_states,
                         tl_past_valid=tl_past_valid,
                         rg_xys=rg_xys,
                         rg_dirs=rg_dirs,
                         rg_types=rg_types)
