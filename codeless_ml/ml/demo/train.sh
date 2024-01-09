# Script demonstrating training and evaluating a classifier for mnist dataset.

FEATURE_SPEC=$(cat <<EOF
feature_spec {
  name: "image"
  fixed_len_feature {
    data_type: TF_DATA_TYPE_FLOAT32
    shape: [784]
  }
}
feature_spec {
  name: "label"
  fixed_len_feature {
    data_type: TF_DATA_TYPE_INT64
    shape: [1]
  }
}
feature_spec {
  name: "width"
  fixed_len_feature {
    data_type: TF_DATA_TYPE_INT64
    shape: [1]
  }
}
feature_spec {
  name: "height"
  fixed_len_feature {
    data_type: TF_DATA_TYPE_INT64
    shape: [1]
  }
}
feature_spec {
  name: "channel"
  fixed_len_feature {
    data_type: TF_DATA_TYPE_INT64
    shape: [1]
  }
}
EOF
)

MAP_CALLABLE=$(cat <<EOF
map_callable {
  closure {
    function_name: "/callable/return_prepare_image_input_fn"
    argument {
      key: "feature_name"
      value {
        string_value: "image"
      }
    }
    argument {
      key: "label_name"
      value {
        string_value: "label"
      }
    }
    argument {
      key: "width_name"
      value {
        string_value: "width"
      }
    }
    argument {
      key: "height_name"
      value {
        string_value: "height"
      }
    }
    argument {
      key: "channel_name"
      value {
        string_value: "channel"
      }
    }
  }
}
EOF
)

CONFIG_FILE=$(mktemp -u)
echo "write config to ${CONFIG_FILE}"
cat >${CONFIG_FILE} <<EOF
user_defined_python_module: ["codeless_ml.ml.register_callable"]
train_dataset {
  tf_record_dataset {
    filename: "$(pwd)/codeless_ml/ml/demo/mnist/train_samples.rio"
    ${FEATURE_SPEC}
  }
  ${MAP_CALLABLE}
  batch_size: 100
  shuffle_buffer_size: 1024
  repeat: 50
}
validation_dataset {
  tf_record_dataset {
    filename: "$(pwd)/codeless_ml/ml/demo/mnist/test_samples.rio"
    ${FEATURE_SPEC}
  }
  ${MAP_CALLABLE}
  batch_size: 100
  shuffle_buffer_size: 1024
  repeat: 1
}
fit_config {
  epochs: 20
  steps_per_epoch: 500
  validation_steps: 100
}
evaluate_config {
  steps: 500 
}
checkpoint_config {
  filepath: "/tmp/mnist_model/cp-{epoch:04d}.ckpt"
}
tensor_board_config {
  log_dir: "/tmp/mnist_model/"
  samples: 20
}
model_config {
  name: "mnist_model"
  description: "simple cnn for minst dataset."
  adadelta_optimizer {
    lr: 0.02 
    rho: 0.95
    epsilon: 1e-7
    decay: 0.0
  }
  loss: LOSS_TYPE_SPARSE_CATEGORICAL_CROSSENTROPY
  metric: [METRIC_TYPE_SPARSE_CATEGORICAL_ACCURACY]
  layer {
    name: "image"
    input {
      shape: [28,28,1]
      dtype: "float32"
      sparse: false
    }
  }
  # layer {
  #   name: "conv1"
  #   conv_2d {
  #     filters: 32
  #     kernel_size: [3, 3]
  #     strides: [1, 1]
  #     padding: PADDING_TYPE_SAME
  #     data_format: DATA_FORMAT_CHANNELS_LAST
  #     activation: ACTIVATION_TYPE_RELU
  #   }
  #   dependency: ["image"]
  # }
  # layer {
  #   name: "conv2"
  #   conv_2d {
  #     filters: 64
  #     kernel_size: [3, 3]
  #     strides: [1, 1]
  #     padding: PADDING_TYPE_SAME
  #     data_format: DATA_FORMAT_CHANNELS_LAST
  #     activation: ACTIVATION_TYPE_RELU
  #   }
  #   dependency: ["conv1"]
  # }
  # layer {
  #   name: "pool2"
  #   max_pooling_2d {
  #     pool_size: [2, 2]
  #     strides: [3, 3]
  #     padding: PADDING_TYPE_SAME
  #     data_format: DATA_FORMAT_CHANNELS_LAST
  #   }
  #   dependency: ["conv2"]
  # }
  # layer {
  #   name: "dropout1"
  #   dropout {
  #     rate: 0.25
  #   }
  #   dependency: ["pool2"]
  # }
  layer {
    name: "flatten"
    flatten {}
    dependency: ["image"]
  }
  layer {
    name: "dense1"
    dense {
      units: 1024
      activation: ACTIVATION_TYPE_RELU
    }
    dependency: ["flatten"]
  }
  # layer {
  #   name: "dropout2"
  #   dropout {
  #     rate: 0.5
  #   }
  #   dependency: ["dense1"]
  # }
  layer {
    name: "class_probabilities"
    dense {
      units: 10
      activation: ACTIVATION_TYPE_SOFTMAX
    }
    dependency: ["dense1"]
    is_output: true
  }
}
EOF

bazel run codeless_ml/ml:train -- \
  --trainer_config_file=${CONFIG_FILE} \
  --job=train
