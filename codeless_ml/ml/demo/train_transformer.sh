MODEL_CONFIG=$(cat <<EOF
model_config {
  name: "nmt_model"
  description: "transformer model for translation"
  ada_optimizer {
    lr {
			custom_schedule {
				d_model: 512
				warmup_steps: 40000
			}
		}
	  beta1: 0.9
		beta2: 0.98
	  epsilon: 1e-9
  }
  loss_config {
    loss_spec {
			custom_loss {
				loss_function_registry_key: "/loss_function/transformer/sparse_cross_entropy"
			}
    }
  }
  metric_config {
    metric_spec {
			custom_metric {
				metric_function_registry_key: "/metric/transformer/accuracy"
			}
    }
  }
  layer {
    name: "pt"
    input {
      shape: [62]
      dtype: "int32"
      sparse: false
    }
  }
  layer {
    name: "en"
    input {
      shape: [58]
      dtype: "int32"
      sparse: false
    }
  }
  layer {
    name: "target_logits"
    output {
      shape: [58, ]
      dtype: 
    }
  }
  layer {
    name: "dense1"
    dense {
      units: 1024
      activation: ACTIVATION_TYPE_RELU
    }
    dependency: ["flatten"]
  }
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
)
