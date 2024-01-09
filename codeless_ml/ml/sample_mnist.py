import os
import numpy as np

import keras
import tensorflow as tf
from absl import app
from absl import logging
from absl import flags
from keras.datasets import mnist
from typing import List

flags.DEFINE_float(
    "downsample_rate", 0.1,
    "The portion of samples to keep for training and testing data.")
flags.DEFINE_string("output_directory", "/tmp/mnist",
                    "Directory holding the sampled result.")

FLAGS = flags.FLAGS


def _float_list_feature(values: List[float]):
  """Returns a float_list from a float / double."""
  # logging.info(values)
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(value: int):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _create_tf_example(x: List[float], y: int) -> tf.train.Example:
  feature = {
      "image": _float_list_feature(x),
      "label": _int64_feature(y),
      "width": _int64_feature(28),
      "height": _int64_feature(28),
      "channel": _int64_feature(1)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _create_tf_record_file(images: np.array, labels: np.array,
                           tf_record_file: str):
  with tf.io.TFRecordWriter(path=tf_record_file) as f:
    for i in range(images.shape[0]):
      x = images[i, :].reshape(-1).tolist()
      y = int(labels[i])
      example = _create_tf_example(x, y)
      f.write(example.SerializeToString())


def main(argv):
  del argv
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  logging.info(x_train.shape)
  logging.info(y_train.shape)
  logging.info(x_test.shape)
  logging.info(y_test.shape)
  num_train = x_train.shape[0]
  train_index = np.arange(num_train)
  np.random.shuffle(train_index)
  train_rand_index = train_index[0:int(num_train * FLAGS.downsample_rate)]
  num_test = x_test.shape[0]
  test_index = np.arange(num_test)
  np.random.shuffle(test_index)
  test_rand_index = test_index[0:int(num_test * FLAGS.downsample_rate)]

  train_x_file = os.path.join(FLAGS.output_directory, "train_x_samples.npy")
  train_y_file = os.path.join(FLAGS.output_directory, "train_y_samples.npy")
  test_x_file = os.path.join(FLAGS.output_directory, "test_x_samples.npy")
  test_y_file = os.path.join(FLAGS.output_directory, "test_y_samples.npy")

  np.save(train_x_file, x_train[train_rand_index, :])
  np.save(train_y_file, y_train[train_rand_index,])
  np.save(test_x_file, x_test[test_rand_index, :])
  np.save(test_y_file, y_test[test_rand_index,])

  # create TFRecord file for the training and validation data.
  train_tf_record_file = os.path.join(FLAGS.output_directory,
                                      "train_samples.rio")
  test_tf_record_file = os.path.join(FLAGS.output_directory, "test_samples.rio")
  _create_tf_record_file(x_train[train_rand_index, :],
                         y_train[train_rand_index,], train_tf_record_file)
  _create_tf_record_file(x_test[test_rand_index, :], y_test[test_rand_index,],
                         test_tf_record_file)


if __name__ == '__main__':
  app.run(main)
