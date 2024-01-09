from typing import Dict, Tuple, TypeVar, List

import tensorflow as tf
import codeless_ml.ml.input_pb2 as input_pb2
import codeless_ml.common.global_variable as gv

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


def _get_tensor_slice_dataset(
    config: input_pb2.DatasetConfig) -> tf.data.Dataset:
  tensors = tuple([
      GVR.retrieve_variable(name)
      for name in config.tensor_slice_dataset.dataset_name
  ])
  return tf.data.Dataset.from_tensor_slices(tensors)


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
            tf.io.FixedLenFeature(shape=tuple(spec.fixed_len_feature.shape),
                                  dtype=tf_data_type_from_proto(
                                      spec.fixed_len_feature.data_type)))
  elif specific_feature == "var_len_feature":
    return (spec.name,
            tf.VarLenFeature(
                dtype=tf_data_type_from_proto(spec.var_len_feature.data_type)))


def _get_tf_record_dataset(config: input_pb2.DatasetConfig) -> tf.data.Dataset:
  tf_config = config.tf_record_dataset
  dataset = tf.data.TFRecordDataset(filenames=tf_config.filename)

  TensorType = TypeVar("TensorType", tf.Tensor, tf.SparseTensor)
  FeatureMapType = Dict[str, TensorType]

  def _parse_function(serialized: bytes) -> FeatureMapType:
    features = dict([_create_feature_spec(f) for f in tf_config.feature_spec])
    return tf.io.parse_single_example(serialized=serialized, features=features)

  return dataset.map(_parse_function)


def get_dataset(config: input_pb2.DatasetConfig) -> tf.data.Dataset:
  if config.HasField("tensor_slice_dataset"):
    dataset = _get_tensor_slice_dataset(config)
  elif config.HasField("tf_record_dataset"):
    dataset = _get_tf_record_dataset(config)
  for map_callable in config.map_callable:
    dataset = dataset.map(GVR.retrieve_callable(map_callable))
  if config.shuffle_buffer_size > 0:
    dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size)
  if config.repeat != 0:
    dataset = dataset.repeat(count=None if config.repeat < 0 else config.repeat)
  if config.batch_size > 0:
    dataset = dataset.batch(config.batch_size)
  return dataset
