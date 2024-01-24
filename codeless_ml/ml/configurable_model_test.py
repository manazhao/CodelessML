import unittest

from absl import logging
from google.protobuf import text_format
import tensorflow as tf
from codeless_ml.ml.configurable_model import ConfigurableModel
from codeless_ml.ml import configurable_model_pb2


class ConfigurableModelTest(unittest.TestCase):

    def setUp(self):
        config_pbtxt = """
      name: "test_configurable_model"
      description: "test constructing keras Model out of configuation"
      layer {
        name: "image_input"
        input {
          shape: [100,100, 3]
          sparse: false
        }
      }
      layer {
        name: "conv1"
        conv_2d {
          # Number of channels.
          filters: 256
          kernel_size: [3]
          strides: [1,1]
          padding: PADDING_TYPE_SAME
          data_format: DATA_FORMAT_CHANNELS_LAST
          use_bias: true
          activation: ACTIVATION_TYPE_RELU
        }
        dependency: {name: "image_input"}
      }
      layer {
        name: "pool1"
        max_pooling_2d {
          pool_size: [2]
          strides: [2]
          padding: PADDING_TYPE_SAME
          data_format: DATA_FORMAT_CHANNELS_LAST
        }
        dependency: {name: "conv1"}
      }
      layer {
        name: "dense1"
        dependency: {name: "pool1"}
        dense {
          units: 10
          activation: ACTIVATION_TYPE_SOFTMAX
          use_bias: false
        }
        is_output: true
      }
      adam_optimizer {
        lr {
            fixed_rate: 0.001
        }
        beta_1: 0.9
        beta_2: 0.999
        weight_decay: 0.001
        epsilon: 1e-7
        amsgrad: false
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
    """
        model_config = configurable_model_pb2.ModelConfig()
        text_format.Parse(config_pbtxt, model_config)
        self._model = ConfigurableModel()
        self._model.init_from_config(model_config)

    def _create_expected_model(self):
        image_input = tf.keras.layers.Input(name="image_input",
                                            shape=(100, 100, 3),
                                            sparse=False)
        conv1 = tf.keras.layers.Conv2D(name="conv1",
                                       filters=256,
                                       kernel_size=3,
                                       strides=(1, 1),
                                       data_format="channels_last",
                                       padding="same",
                                       use_bias=True,
                                       activation="relu")(image_input)
        pool1 = tf.keras.layers.MaxPooling2D(
            name="pool1",
            pool_size=2,
            strides=2,
            padding="same",
            data_format="channels_last")(conv1)
        dense1 = tf.keras.layers.Dense(name="dense1",
                                       units=10,
                                       activation="softmax",
                                       use_bias=False)(pool1)
        return tf.keras.models.Model(inputs=[image_input],
                                     outputs=[dense1],
                                     name="test_configurable_model")

    def testCreateModel(self):
        model_json = self._model.model.to_json()
        expected_model_json = self._create_expected_model().to_json()
        self.assertEqual(model_json, expected_model_json)


if __name__ == '__main__':
    unittest.main()
