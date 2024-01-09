import unittest
import os

from absl import logging
from google.protobuf import text_format
from parameterized import parameterized
from typing import List
import numpy as np
import tensorflow as tf

import codeless_ml.common.global_variable as gv
import codeless_ml.ml.train_pb2 as trainer_pb2
from codeless_ml.ml.model_trainer import ModelTrainer

_TRAINER_CONFIG_PBTXT = """
  user_defined_python_module: ["codeless_ml.ml.setup_mnist"]
  train_dataset {
    tensor_slice_dataset {
      dataset_name: ["/mnist_train_x", "/mnist_train_y"]
    }
    batch_size: 10
    shuffle_buffer_size: 1024
    repeat: 1
  }
  validation_dataset {
    tensor_slice_dataset {
      dataset_name: ["/mnist_validation_x", "/mnist_validation_y"]
    }
    batch_size: 10
    shuffle_buffer_size: 1024
    repeat: 1
  }
  evaluation_dataset {
    tensor_slice_dataset {
      dataset_name: ["/mnist_validation_x", "/mnist_validation_y"]
    }
    batch_size: 10
    shuffle_buffer_size: 1024
    repeat: 1
  }
  fit_config {
    epochs: 1
    steps_per_epoch: 100
    validation_steps: 100
  }
  evaluate_config {
    steps: 100 
  }
  save_model_config {
    output_directory: ""  
  }
  model_config {
    name: "mnist_model"
    description: "simple cnn for minst dataset."
    adadelta_optimizer {
      lr: 1.0
      rho: 0.95
      epsilon: 1e-7
      weight_decay: 0.0
    }
    loss_config {
      loss_spec {
        standard_loss: LOSS_TYPE_CATEGORICAL_CROSSENTROPY
      }
    }
    metric_config {
      metric_spec {
        standard_metric: METRIC_TYPE_CATEGORICAL_ACCURACY
      }
    }
    layer {
      name: "input"
      input {
        shape: [28,28,1]
        dtype: "float32"
        sparse: false
      }
    }
    layer {
      name: "conv1"
      conv_2d {
        filters: 32
        kernel_size: [3, 3]
        strides: [1, 1]
        padding: PADDING_TYPE_SAME
        data_format: DATA_FORMAT_CHANNELS_LAST
        activation: ACTIVATION_TYPE_RELU
      }
      dependency: ["input"]
    }
    layer {
      name: "conv2"
      conv_2d {
        filters: 64
        kernel_size: [3, 3]
        strides: [1, 1]
        padding: PADDING_TYPE_SAME
        data_format: DATA_FORMAT_CHANNELS_LAST
        activation: ACTIVATION_TYPE_RELU
      }
      dependency: ["conv1"]
    }
    layer {
      name: "pool2"
      max_pooling_2d {
        pool_size: [2, 2]
        strides: [3, 3]
        padding: PADDING_TYPE_SAME
        data_format: DATA_FORMAT_CHANNELS_LAST
      }
      dependency: ["conv2"]
    }
    layer {
      name: "dropout1"
      dropout {
        rate: 0.25
      }
      dependency: ["pool2"]
    }
    layer {
      name: "flatten"
      flatten {}
      dependency: ["dropout1"]
    }
    layer {
      name: "dense1"
      dense {
        units: 128
        activation: ACTIVATION_TYPE_RELU
      }
      dependency: ["flatten"]
    }
    layer {
      name: "dropout2"
      dropout {
        rate: 0.5
      }
      dependency: ["dense1"]
    }
    layer {
      name: "output"
      dense {
        units: 10
        activation: ACTIVATION_TYPE_SOFTMAX
      }
      dependency: ["dropout2"]
      is_output: true
    }
  }
"""

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


class ModelTrainerTest(unittest.TestCase):

  def setUp(self):
    self._trainer_config = trainer_pb2.ModelTrainerConfig()
    text_format.Parse(_TRAINER_CONFIG_PBTXT, self._trainer_config)
    output_directory = os.path.join(os.environ["TEST_TMPDIR"], "mnist_model")
    if not os.path.exists(output_directory):
      os.mkdir(output_directory)
    self._trainer_config.save_model_config.output_directory = output_directory
    self._trainer = ModelTrainer()
    self._trainer.init_from_config(trainer_config=self._trainer_config)

  def _test_model_from_full_model_file(self, pred_x, full_model_file: str,
                                       expected_pred_result: List[float]):
    tmp_trainer_config_pbtxt = """
        load_model_config {
          model_path: "%s"
        }
    """ % (full_model_file)
    self._load_model_and_test(pred_x, tmp_trainer_config_pbtxt,
                              expected_pred_result)

  def _load_model_and_test(self, pred_x: np.array, trainer_config_pbtxt: str,
                           expected_pred_result: List[float]):
    tmp_trainer_config = trainer_pb2.ModelTrainerConfig()
    text_format.Parse(trainer_config_pbtxt, tmp_trainer_config)
    tmp_trainer = ModelTrainer()
    tmp_trainer.init_from_config(tmp_trainer_config)
    expected_pred = tmp_trainer.configurable_model.model.predict(x=pred_x)
    np.testing.assert_array_equal(expected_pred_result, expected_pred)

  def testTrainAndEvaluate(self):
    self._trainer.train()
    expected_eval_result = self._trainer.evaluate()
    saved_model_path = self._trainer.save_model()
    tmp_trainer_config = trainer_pb2.ModelTrainerConfig()
    # Setup the evaluation dataset and evaluation config.
    tmp_trainer_config.evaluation_dataset.CopyFrom(
        self._trainer_config.evaluation_dataset)
    tmp_trainer_config.evaluate_config.CopyFrom(
        self._trainer_config.evaluate_config)
    # Configure to load the full model.
    tmp_trainer_config.load_model_config.model_path = saved_model_path
    tmp_trainer = ModelTrainer()
    tmp_trainer.init_from_config(tmp_trainer_config)
    eval_result = tmp_trainer.evaluate()
    np.testing.assert_almost_equal(expected_eval_result, eval_result)

  def testSaveAndLoadModel(self):
    self._trainer.train()
    pred_x = np.copy(GVR.retrieve_variable("/mnist_validation_x")[0,]).reshape(
        (-1, 28, 28, 1))
    # Make prediction for pred_x with current model.
    expected_pred_result = self._trainer.configurable_model.model.predict(
        x=pred_x)
    # Now load the saved model and make prediction with it.
    self._test_model_from_full_model_file(pred_x, self._trainer.save_model(),
                                          expected_pred_result)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  unittest.main()
