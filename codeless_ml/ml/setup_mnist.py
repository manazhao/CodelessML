import os

from absl import logging
import numpy as np
import tensorflow as tf

import codeless_ml.common.global_variable as gv

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


def _setup_minst():
  img_rows, img_cols = 28, 28
  mnist_root = os.path.join(os.environ["TEST_SRCDIR"],
                            os.environ["TEST_WORKSPACE"],
                            "codeless_ml/ml/testdata/mnist")
  logging.info("mnist data dir: %s" % (mnist_root))
  train_x = np.load(os.path.join(mnist_root, "train_x_samples.npy"))
  train_y = np.load(os.path.join(mnist_root, "train_y_samples.npy"))
  test_x = np.load(os.path.join(mnist_root, "test_x_samples.npy"))
  test_y = np.load(os.path.join(mnist_root, "test_y_samples.npy"))
  train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols,
                            1).astype("float32")
  test_x = test_x.reshape(test_x.shape[0], img_rows, img_cols,
                          1).astype("float32")
  train_x /= 255
  test_x /= 255
  num_classes = 10
  train_y = tf.keras.utils.to_categorical(train_y, num_classes)
  test_y = tf.keras.utils.to_categorical(test_y, num_classes)
  GVR.register_variable("/mnist_train_x", train_x)
  GVR.register_variable("/mnist_train_y", train_y)
  GVR.register_variable("/mnist_validation_x", test_x)
  GVR.register_variable("/mnist_validation_y", test_y)


logging.info("setup mnist datasets.")
_setup_minst()
