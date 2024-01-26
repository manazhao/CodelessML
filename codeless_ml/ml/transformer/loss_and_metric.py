import tensorflow as tf

from keras.utils import register_keras_serializable

import codeless_ml.common.global_variable as gv

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


@register_keras_serializable(package="codeless_ml.ml.transformer")
def _loss_function(label, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.math.logical_not(tf.math.equal(label, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


@register_keras_serializable(package="codeless_ml.ml.transformer")
def _accuracy_function(label, pred):
    # batch_size = tf.shape(label)[0]
    # seq_len = tf.shape(label)[1]
    # tf.ensure_shape(label, [batch_size, seq_len])
    # tf.ensure_shape(pred, [batch_size, seq_len, None])
    label = tf.cast(label, tf.int64)
    accuracies = tf.equal(label, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(label, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


LOSS_REGESTRY_KEY = "/loss_function/transformer/sparse_cross_entropy"
METRIC_REGISTRY_KEY = "/metric/transformer/accuracy"

GVR.register_callable(LOSS_REGESTRY_KEY, _loss_function)
GVR.register_callable(METRIC_REGISTRY_KEY, _accuracy_function)
