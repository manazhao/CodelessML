import tensorflow as tf

from typing import Tuple


def agent_centric_transformation(
        xys: tf.Tensor, yaws: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Get the rotation and translation paramteters for each agent in xys so that
    the new coordinate frame centers at each agent with its x-axis pointing to
    the agent's bbox yaws.
    """
    tf.ensure_shape(xys, [None, 2])
    tf.ensure_shape(yaws, [None, 1])
    assert xys.shape[0] == yaws.shape[
        0], "xys and yaws should have the same number of agents."

    cos_vals, sin_vals = tf.math.cos(yaws), tf.math.sin(yaws)
    rotation = tf.reshape(
        tf.concat([cos_vals, sin_vals, -sin_vals, cos_vals], axis=1),
        [-1, 2, 2])
    tf.ensure_shape(rotation, [None, 2, 2])
    translation = -tf.linalg.matmul(rotation, xys, transpose_b=True)
    translation = tf.transpose(translation, perm=[0, 2, 1])  # shape: (N, N, 2)
    #split translation matrix's x and ys
    translation_xs, translation_ys = tf.split(translation, 2, axis=2)
    # tf.print(translation_xs.shape)
    # tf.print(translation_ys.shape)
    translation_xs = tf.squeeze(translation_xs)
    translation_ys = tf.squeeze(translation_ys)
    translation_xs = tf.expand_dims(tf.linalg.diag_part(translation_xs),
                                    axis=1)
    translation_ys = tf.expand_dims(tf.linalg.diag_part(translation_ys),
                                    axis=1)
    translation = tf.concat([translation_xs, translation_ys], axis=1)
    return rotation, translation


def transform_position(rotation: tf.Tensor, translation: tf.Tensor,
                       xys: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # rotation shape: (M, 2, 2)
    tf.ensure_shape(rotation, [None, 2, 2])
    # translation shape: (M, 2)
    tf.ensure_shape(translation, [None, 2])
    # xys shape: [N, 2]
    tf.ensure_shape(xys, [None, 2])
    translation = tf.expand_dims(translation, axis=1)
    # shape: [M, N ,2]
    return tf.transpose(tf.linalg.matmul(rotation, xys, transpose_b=True),
                        perm=[0, 2, 1]) + translation


def yaws_to_vecs(yaws: tf.Tensor):
    # shape: [N, 1]
    tf.ensure_shape(yaws, [None, 1])
    rows = yaws.shape[0]
    # shape: [N, 2]
    return tf.concat([tf.math.cos(yaws), tf.math.sin(yaws)], axis=1)


def transform_direction(rotation: tf.Tensor, dirs: tf.Tensor):
    tf.ensure_shape(dirs, [None, 2])
    return tf.transpose(tf.linalg.matmul(rotation, dirs, transpose_b=True),
                        perm=[0, 2, 1])


def transform_yaws(rotation: tf.Tensor, yaws: tf.Tensor):
    return transform_direction(rotation, yaws_to_vecs(yaws))


def transform_multi_timestep_positions(rotation, translation, x):
    tf.ensure_shape(x, [None, None, 2])
    num_rows, num_time_steps = x.shape[0], x.shape[1]
    # shape: [num_rows * num_time_steps, 2]
    x = tf.reshape(x, [-1, 2])
    x = transform_position(rotation, translation, x)
    # shape: [M, N, T, 2]
    # where M is the number of rows of rotation
    # and N is the number of rows of x.
    return tf.reshape(x, [rotation.shape[0], num_rows, -1, 2])


def transform_multi_timestep_dirs(rotation: tf.Tensor, x):
    tf.ensure_shape(x, [None, None, 2])
    num_rows, num_time_steps = x.shape[0], x.shape[1]
    x = tf.reshape(x, [-1, 2])
    # shape: [M, N, T, 2]
    # where M is the number of rows of rotation and N is the number of rows of x.
    return tf.reshape(transform_direction(rotation, x),
                      [rotation.shape[0], num_rows, -1, 2])


def transform_multi_timestep_yaws_to_vecs(x):
    tf.ensure_shape(x, [None, None, 1])
    rows = x.shape[0]
    x = tf.reshape(x, [-1, 1])
    # shape: [N, T, 2]
    return tf.reshape(yaws_to_vecs(x), [rows, -1, 2])
