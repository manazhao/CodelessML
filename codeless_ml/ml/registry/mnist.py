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
        width = feature_map.get(width_name)[0][0]
        height = feature_map.get(height_name)[0][0]
        channel = feature_map.get(channel_name)[0][0]
        if (width is not None and height is not None and channel is not None):
            feature_tensor = tf.reshape(feature_tensor,
                                        shape=(-1, width, height, channel))
        return feature_tensor, label_tensor

    return _input_fn


_FUNCTION_NAME = "/callable/return_prepare_image_input_fn"
logging.info("Register callable: %s" % (_FUNCTION_NAME))
GVR.register_callable(_FUNCTION_NAME, return_prepare_image_input_fn)
