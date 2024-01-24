from absl import logging
import tensorflow as tf
from typing import Any, Callable, Tuple, Mapping, TypeVar

import codeless_ml.common.global_variable as gv

GVR = gv.GLOBAL_VARIABLE_REPOSITORY

TensorType = TypeVar("TensorType", tf.Tensor, tf.SparseTensor)


def return_prepare_image_input_fn(feature_name: str,
                                  label_name: str,
                                  width_name=None,
                                  height_name=None,
                                  channel_name=None) -> Callable[..., Any]:

    def _input_fn(
        feature_map: Mapping[str,
                             TensorType]) -> Tuple[TensorType, TensorType]:
        feature_tensor = feature_map.get(feature_name)
        label_tensor = feature_map.get(label_name)
        if feature_tensor is None or label_tensor is None:
            logging.fatal(
                "feature_tensor and label_tensor must exist in the feature map."
            )
        width_tensor = tf.reshape(feature_map.get(width_name), (1, ))
        height_tensor = tf.reshape(feature_map.get(height_name), (1, ))
        channel_tensor = tf.reshape(feature_map.get(channel_name), (1, ))
        if (width_tensor is not None and height_tensor is not None
                and channel_tensor is not None):
            feature_tensor = tf.reshape(
                feature_tensor,
                shape=tf.concat([width_tensor, height_tensor, channel_tensor],
                                axis=0))
        return feature_tensor, label_tensor

    return _input_fn


_FUNCTION_NAME = "/callable/return_prepare_image_input_fn"
logging.info("Register callable: %s" % (_FUNCTION_NAME))
GVR.register_callable(_FUNCTION_NAME, return_prepare_image_input_fn)
