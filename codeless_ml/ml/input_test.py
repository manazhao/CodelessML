import os
import unittest

import numpy as np
import tensorflow as tf
import google.protobuf.text_format as text_format

import codeless_ml.ml.input_pb2 as input_pb2
import codeless_ml.ml.input as data_input
import codeless_ml.common.global_variable as gv

from pathlib import Path

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy(
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class InputTest(unittest.TestCase):

    def testTensorSliceDataset(self):
        dataset_config = input_pb2.DatasetConfig()
        tensor_name = "/input/tensor"
        tensor = np.array([[1., 2.], [3., 4.], [5., 6.]])
        GVR.register_variable(tensor_name, tensor)
        dataset_config.tensor_slice_dataset.dataset_name.append(tensor_name)
        dataset_config.batch_size = 3
        dataset_config.repeat = 10

        def _decrease_by_one(x):
            return x - 1

        map_fn_name = "/callable/decrease_by_one"
        GVR.register_callable(map_fn_name, _decrease_by_one)
        map_callable = dataset_config.map_callable.add()
        map_callable.function_name = map_fn_name
        dataset = data_input.get_dataset(dataset_config)
        expected_tensor = np.array([[0., 1.], [2., 3.], [4., 5.]])
        for element in dataset:
            np.testing.assert_array_equal(element.numpy(), expected_tensor)

    def _create_tf_example(self, str_val: bytes, float_val: float,
                           int_val: int) -> tf.train.Example:
        feature = {
            "str_feat": _bytes_feature(str_val),
            "float_feat": _float_feature(float_val),
            "int_feat": _int64_feature(int_val)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def testTfRecordDataset(self):
        example1 = self._create_tf_example(str_val=b"hello",
                                           float_val=1.0,
                                           int_val=1)
        example2 = self._create_tf_example(str_val=b"tensorflow",
                                           float_val=2.0,
                                           int_val=2)
        example3 = self._create_tf_example(str_val=b"cool",
                                           float_val=3.0,
                                           int_val=3)
        examples = [example1, example2, example3]
        tf_record_file = os.path.join(os.environ["TEST_TMPDIR"],
                                      "test.tfrecord")
        with tf.io.TFRecordWriter(path=tf_record_file) as f:
            for e in examples:
                f.write(e.SerializeToString())
        dataset_config = input_pb2.DatasetConfig()
        dataset_config.tf_record_dataset.filename = tf_record_file

        feature_spec = dataset_config.tf_record_dataset.feature_spec.add()
        feature_spec.name = "str_feat"
        feature_spec.fixed_len_feature.data_type = input_pb2.TF_DATA_TYPE_STRING
        feature_spec.fixed_len_feature.shape.extend([1])

        feature_spec = dataset_config.tf_record_dataset.feature_spec.add()
        feature_spec.name = "float_feat"
        feature_spec.fixed_len_feature.data_type = input_pb2.TF_DATA_TYPE_FLOAT32
        feature_spec.fixed_len_feature.shape.extend([1])

        feature_spec = dataset_config.tf_record_dataset.feature_spec.add()
        feature_spec.name = "int_feat"
        feature_spec.fixed_len_feature.data_type = input_pb2.TF_DATA_TYPE_INT64
        feature_spec.fixed_len_feature.shape.extend([1])

        def _mutate_map_entry(x):
            x["str_feat"] = tf.strings.substr(input=x["str_feat"],
                                              pos=0,
                                              len=3)
            x["int_feat"] = x["int_feat"] * 2
            x["float_feat"] = x["float_feat"] * 4
            return x

        map_fn_name = "/callable/mutate_map_entry"
        GVR.register_callable("/callable/mutate_map_entry", _mutate_map_entry)
        map_callable = dataset_config.map_callable.add()
        map_callable.function_name = map_fn_name

        dataset_config.batch_size = 3
        dataset_config.repeat = 10
        dataset = data_input.get_dataset(dataset_config)
        for feature_map in dataset:
            np.testing.assert_array_equal(
                feature_map["str_feat"].numpy(),
                np.array([[b"hel"], [b"ten"], [b"coo"]]))
            np.testing.assert_array_equal(feature_map["int_feat"].numpy(),
                                          np.array([[2], [4], [6]]))
            np.testing.assert_array_equal(feature_map["float_feat"].numpy(),
                                          np.array([[4.], [8.], [12.]]))


if __name__ == '__main__':
    unittest.main()
