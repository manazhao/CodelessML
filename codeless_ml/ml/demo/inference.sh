#!/bin/bash

# First create a SavedModel from a checkpoint file.
# The resulting SavedModel will be automatically versioned by timestamp.
bazel run codeless_ml/ml:saved_model_converter_main  -- \
--checkpoint_path=/tmp/mnist_model/cp-0005.ckpt
--saved_model_path=/tmp/mnist_model/saved_model

# Test the serving using saved_model_cli tool.
saved_model_cli run \
  --dir=/tmp/mnist_model/saved_model/1557728211 \
  --tag_set serve \
  --signature_def serving_default \
  --input_exp='input_image=np.random.rand(1,28,28,3)'
