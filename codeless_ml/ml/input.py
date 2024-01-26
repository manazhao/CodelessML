import tensorflow as tf
import tensorflow
import tensorflow_datasets as tfds

import codeless_ml.ml.input_pb2 as input_pb2
import codeless_ml.common.global_variable as gv

from typing import Dict, Tuple, TypeVar, List
from absl import logging

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


def _get_tensor_slice_dataset(
        config: input_pb2.DatasetConfig) -> tf.data.Dataset:
    tensors = tuple([
        GVR.retrieve_variable(name)
        for name in config.tensor_slice_dataset.dataset_name
    ])
    return tf.data.Dataset.from_tensor_slices(tensors)


def _get_tfds(config: input_pb2.DatasetConfig) -> tf.data.Dataset:
    batch_size = config.batch_size

    tfds_config = config.tfds
    if tfds_config.data_dir == '':
        logging.fatal("data_dir is required.")
        return None

    assert tfds_config.split, "split must be specified"
    return tfds.load(tfds_config.name,
                     split=tfds_config.split,
                     data_dir=tfds_config.data_dir,
                     shuffle_files=tfds_config.shuffle_files,
                     download=False,
                     as_supervised=True,
                     with_info=False)


def tf_data_type_from_proto(t: input_pb2.TfDataType) -> tf.DType:
    if t == input_pb2.TF_DATA_TYPE_INT32:
        return tf.int32
    elif t == input_pb2.TF_DATA_TYPE_INT64:
        return tf.int64
    elif t == input_pb2.TF_DATA_TYPE_FLOAT32:
        return tf.float32
    elif t == input_pb2.TF_DATA_TYPE_FLOAT64:
        return tf.float64
    elif t == input_pb2.TF_DATA_TYPE_STRING:
        return tf.string


TfFeatureType = TypeVar("TfFeatureType", tf.io.FixedLenFeature,
                        tf.io.VarLenFeature)


def _create_feature_spec(
        spec: input_pb2.FeatureSpec) -> Tuple[str, TfFeatureType]:
    specific_feature = spec.WhichOneof("specific_feature")
    if specific_feature == "fixed_len_feature":
        return (spec.name,
                tf.io.FixedLenFeature(shape=tuple(
                    spec.fixed_len_feature.shape),
                                      dtype=tf_data_type_from_proto(
                                          spec.fixed_len_feature.data_type)))
    elif specific_feature == "var_len_feature":
        return (
            spec.name,
            tf.VarLenFeature(
                dtype=tf_data_type_from_proto(spec.var_len_feature.data_type)))


def _get_tf_record_dataset(config: input_pb2.DatasetConfig) -> tf.data.Dataset:
    tf_config = config.tf_record_dataset
    dataset = tf.data.TFRecordDataset(filenames=tf_config.filename)

    def _parse_function(serialized: bytes):
        features = dict(
            [_create_feature_spec(f) for f in tf_config.feature_spec])
        return tf.io.parse_single_example(serialized=serialized,
                                          features=features)

    return dataset.map(_parse_function)


def get_dataset(config: input_pb2.DatasetConfig) -> tf.data.Dataset:
    dataset = None
    if config.HasField("tensor_slice_dataset"):
        dataset = _get_tensor_slice_dataset(config)
    elif config.HasField("tf_record_dataset"):
        dataset = _get_tf_record_dataset(config)
    elif config.HasField("tfds"):
        dataset = _get_tfds(config)
    else:
        logging.fatal("dataset is required")

    if config.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size)

    for map_callable in config.pre_batch_map_callable:
        dataset = dataset.map(GVR.retrieve_callable(map_callable),
                              tf.data.AUTOTUNE)
    if config.batch_size > 0:
        dataset = dataset.batch(config.batch_size)

    for map_callable in config.map_callable:
        dataset = dataset.map(GVR.retrieve_callable(map_callable),
                              tf.data.AUTOTUNE)
    if config.repeat != 0:
        dataset = dataset.repeat(
            count=None if config.repeat < 0 else config.repeat)
    return dataset
