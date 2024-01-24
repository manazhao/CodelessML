import unittest
import os

from absl import logging
from google.protobuf import text_format
from typing import List
import numpy as np
import tensorflow as tf

import codeless_ml.common.global_variable as gv
import codeless_ml.ml.input as data_input
import codeless_ml.ml.train_pb2 as trainer_pb2

from codeless_ml.ml.transformer.schedule import CREATE_SCHEDULE_REGESTRY_KEY
from codeless_ml.ml.model_trainer import ModelTrainer

_num_samples = 100
_pt_vocab_size = 20
_en_vocab_size = 10
_pt_seq_len = 32
_en_seq_len = 16
_d_model = 512
_batch_size = 10

_train_dataset_name = "/transformer_model_test/train"
_validation_dataset_name = "/transformer_model_test/validation"
_test_dataset_name = "/transformer_model_test/test"


def _create_tensors(dataset_name_prefix: str):
    pt_tensors = tf.random.uniform([_num_samples, _pt_seq_len],
                                   1,
                                   _pt_vocab_size,
                                   dtype=tf.dtypes.int32)
    en_tensors = tf.random.uniform([_num_samples, _en_seq_len],
                                   1,
                                   _en_vocab_size,
                                   dtype=tf.dtypes.int32)
    label_tensors = tf.random.uniform([_num_samples, _en_seq_len],
                                      1,
                                      _en_vocab_size,
                                      dtype=tf.dtypes.int32)
    gv.GLOBAL_VARIABLE_REPOSITORY.register_variable(
        f"{dataset_name_prefix}_pt", pt_tensors)
    gv.GLOBAL_VARIABLE_REPOSITORY.register_variable(
        f"{dataset_name_prefix}_en", en_tensors)
    gv.GLOBAL_VARIABLE_REPOSITORY.register_variable(
        f"{dataset_name_prefix}_labels", label_tensors)


def _register_dataset():
    train_tensors = _create_tensors(_train_dataset_name)
    validation_tensors = _create_tensors(_validation_dataset_name)
    test_tensors = _create_tensors(_test_dataset_name)


_register_dataset()

_input_map_fn_name = "/transformer_model_test/input_map"


def _input_map_fn(pt, en, labels):
    return {"en": en, "pt": pt}, labels


gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(_input_map_fn_name,
                                                _input_map_fn)

_TRAINER_CONFIG_PBTXT = f"""
user_defined_python_module: [
  "codeless_ml.ml.transformer.schedule",
  "codeless_ml.ml.transformer.loss_and_metric"
]
train_dataset {{
  tensor_slice_dataset {{
    dataset_name: [
      "{_train_dataset_name}_pt",
      "{_train_dataset_name}_en",
      "{_train_dataset_name}_labels"
    ]
  }}
  batch_size: {_batch_size}
  shuffle_buffer_size: 1024
  repeat: 1
  map_callable {{
    function_name: "{_input_map_fn_name}"
  }}
}}
validation_dataset {{
  tensor_slice_dataset {{
    dataset_name: [
      "{_validation_dataset_name}_pt",
      "{_validation_dataset_name}_en",
      "{_validation_dataset_name}_labels"
    ]
  }}
  batch_size: {_batch_size}
  shuffle_buffer_size: 1024
  repeat: 1
  map_callable {{
    function_name: "{_input_map_fn_name}"
  }}
}}
evaluation_dataset {{
  tensor_slice_dataset {{
    dataset_name: [
      "{_test_dataset_name}_pt",
      "{_test_dataset_name}_en",
      "{_test_dataset_name}_labels"
    ]
  }}
  batch_size: {_batch_size}
  shuffle_buffer_size: 1024
  repeat: 1
  map_callable {{
    function_name: "{_input_map_fn_name}"
  }}
}}
fit_config {{
  epochs: 1
  steps_per_epoch: 100
  validation_steps: 100
}}
evaluate_config {{
  steps: 100 
}}
save_model_config {{
  output_directory: ""  
}}
model_config {{
  name: "nmt_model"
  description: "transformer model for translation"
  adam_optimizer {{
    lr {{
      custom_schedule {{
        closure {{
          function_name: "{CREATE_SCHEDULE_REGESTRY_KEY}"
          argument {{
            key: "d_model"
            value {{
              int32_value: {_d_model}
            }}
          }}
          argument {{
            key: "warmup_steps"
            value {{
              int32_value: 1000
            }}
          }}
        }}
      }}
    }}
    beta_1: 0.9
    beta_2: 0.98
    epsilon: 1e-9
  }}
  loss_config {{
    loss_spec {{
      custom_loss {{
        function_name: "/loss_function/transformer/sparse_cross_entropy"
      }}
    }}
   }}
  metric_config {{
    metric_spec {{
      custom_metric {{
        function_name: "/metric/transformer/accuracy"
      }}
    }}
  }}
  layer {{
    name: "pt"
    input {{
      shape: [{_pt_seq_len}]
      dtype: "int32"
      sparse: false
    }}
  }}
  layer {{
    name: "en"
    input {{
      shape: [{_en_seq_len}]
      dtype: "int32"
      sparse: false
    }}
  }}
  layer {{
    name: "encoder"
    transformer_encoder {{
      num_layers: 4
      d_model: {_d_model}
      num_heads: 2
      dff: 512
      vocab_size: {_pt_vocab_size}
      dropout_rate: 0.1
    }}
    dependency {{ name: "pt" }}
  }}
  layer {{
    name: "decoder"
    transformer_decoder {{
      num_layers: 4
      d_model: {_d_model}
      num_heads: 2
      dff: 512
      vocab_size: {_en_vocab_size}
      dropout_rate: 0.1
    }}
    dependency {{ name: "en" }}
    dependency {{ name: "encoder" }}
  }}
  layer {{
    name: "target"
    dense {{
      units: {_en_vocab_size}
      activation: ACTIVATION_TYPE_LINEAR
    }}
    is_output: true
    dependency {{ name: "decoder" }}
  }}
}}
"""


class ModelTrainerTest(unittest.TestCase):

    def setUp(self):
        self._trainer_config = trainer_pb2.ModelTrainerConfig()
        text_format.Parse(_TRAINER_CONFIG_PBTXT, self._trainer_config)
        output_directory = os.path.join(os.environ["TEST_TMPDIR"],
                                        "transformer_model")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self._trainer_config.save_model_config.output_directory = output_directory
        self._trainer = ModelTrainer()
        self._trainer.init_from_config(trainer_config=self._trainer_config)

    def test_dataset(self):
        for named_features, labels in data_input.get_dataset(
                self._trainer._trainer_config.train_dataset):
            pt = named_features["pt"]
            en = named_features["en"]
            self.assertEqual(pt.shape, [_batch_size, _pt_seq_len])
            self.assertEqual(en.shape, [_batch_size, _en_seq_len])
            self.assertEqual(labels.shape, [_batch_size, _en_seq_len])
            break

    def test_train_and_evaluate(self):
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
        np.testing.assert_almost_equal(expected_eval_result,
                                       eval_result,
                                       decimal=5)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    unittest.main()
