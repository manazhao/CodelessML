import unittest

from absl import logging
from google.protobuf import text_format
import tensorflow as tf
from codeless_ml.ml.configurable_model import ConfigurableModel
from codeless_ml.ml import configurable_model_pb2


class EmbeddingModelTest(unittest.TestCase):

    def setUp(self):
        config_pbtxt = """
      name: "embedding_model_test"
      description: "test embedding layers"
      layer {
        name: "text_input"
        input {
          shape: [1]
          sparse: false
          dtype: "string"
        }
      }
      layer {
        name: "text_vectorization"
        text_vectorization {
          max_tokens: 10
          vocab: ["good", "bad", "movie", "superb", "recommend"]
        }
        dependency: {name: "text_input"}
      }
      layer {
        name: "embedding"
        embedding {
          input_dim: 100
          output_dim: 16
        }
        dependency: {name: "text_vectorization"}
      }
      layer {
        name: "average_embedding"
        global_average_pooling_1d {}
        dependency: {name: "embedding"}
      }
      layer {
        name: "dense"
        dense {
          units: 5
          activation: ACTIVATION_TYPE_SOFTMAX
          use_bias: false
        }
        dependency: {name: "average_embedding"}
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
          standard_loss:LOSS_TYPE_SPARSE_CATEGORICAL_CROSSENTROPY
        }
      }
      metric_config {
        metric_spec {
          standard_metric: METRIC_TYPE_SPARSE_CATEGORICAL_ACCURACY
        }
      }
    """
        model_config = configurable_model_pb2.ModelConfig()
        text_format.Parse(config_pbtxt, model_config)
        self._model = ConfigurableModel()
        self._model.init_from_config(model_config)
        self._model.model.summary()

    def testDefaultStandardize(self):
        self._model.model.predict(tf.constant([["good movie"], ["bad"]]))


if __name__ == '__main__':
    unittest.main()
