from absl import logging
from google.protobuf import text_format

import tensorflow as tf

import codeless_ml.ml.configurable_model_pb2 as configurable_model_pb2
import codeless_ml.common.global_variable as gv

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


def _non_empty_string_or_none(value):
  return None if not value else value


def _padding_type_to_string(padding_type):
  if padding_type == configurable_model_pb2.PADDING_TYPE_VALID:
    return "valid"
  elif padding_type == configurable_model_pb2.PADDING_TYPE_SAME:
    return "same"
  raise ValueError("invalid padding_type %s" % (padding_type))


def _data_format_to_string(data_format):
  if data_format == configurable_model_pb2.DATA_FORMAT_CHANNELS_FIRST:
    return "channels_first"
  elif data_format == configurable_model_pb2.DATA_FORMAT_CHANNELS_LAST:
    return "channels_last"
  else:
    raise ValueError("invalid data format: %s" % (data_format))


def _activation_type_to_string(activation_type):
  if activation_type == configurable_model_pb2.ACTIVATION_TYPE_TAHN:
    return "tahn"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_SOFTMAX:
    return "softmax"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_ELU:
    return "elu"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_SELU:
    return "selu"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_SOFTPLUS:
    return "softplus"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_SOFTSIGN:
    return "softsign"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_RELU:
    return "relu"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_SIGMOID:
    return "sigmoid"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_HARD_SIGMOID:
    return "hard_sigmoid"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_EXPONENTIAL:
    return "exponential"
  elif activation_type == configurable_model_pb2.ACTIVATION_TYPE_LINEAR:
    return "linear"
  else:
    raise ValueError("invalid activation_type %s" % (activation_type))


def _create_sgd_optimizer(config):
  return tf.keras.optimizers.SGD(lr=config.lr,
                                 momentum=config.momentum,
                                 weight_decay=config.weight_decay,
                                 nesterov=config.nesterov)


def _create_rmsprop_optimizer(config):
  return tf.keras.optimizers.RMSprop(lr=config.lr,
                                     rho=config.rho,
                                     epsilon=config.epsilon,
                                     weight_decay=config.weight_decay)


def _create_adagrad_optimizer(config):
  return tf.keras.optimizers.Adagrad(lr=config.lr,
                                     epsilon=config.epsilon,
                                     weight_decay=config.weight_decay)


def _create_adadelta_optimizer(config):
  return tf.keras.optimizers.Adadelta(lr=config.lr,
                                      rho=config.rho,
                                      epsilon=config.epsilon,
                                      weight_decay=config.weight_decay)


def _create_adam_optimizer(config):
  return tf.keras.optimizers.Adam(lr=config.lr,
                                  beta_1=config.beta_1,
                                  beta_2=config.beta_2,
                                  epsilon=config.epsilon,
                                  weight_decay=config.weight_decay,
                                  amsgrad=config.amsgrad)


class ConfigurableModel(object):

  def __init__(self):
    self._model_config = configurable_model_pb2.ModelConfig()
    self._inputs = []
    self._outputs = []
    self._tensor_lookup = {}
    self._name_to_layer_config = {}
    self._model = None

  @property
  def model(self):
    return self._model

  @model.setter
  def model(self, value: tf.keras.models.Model):
    self._model = value

  def _create_loss(self):
    loss_config = self._model_config.loss_config
    loss_list = []
    named_loss = True
    for loss_spec in loss_config.loss_spec:
      loss = None
      if loss_spec.HasField("standard_loss"):
        loss = configurable_model_pb2.LossType.Name(
            loss_spec.standard_loss)[len("LOSS_TYPE_"):].lower()
      elif loss_spec.HasField("custom_loss"):
        loss = GVR.retrieve_callable(
            loss_spec.custom_loss.loss_function_registry_key)
      if loss is not None:
        loss_list.append((loss_spec.name, loss))
        named_loss = (named_loss and loss_spec.name)

    return dict(loss_list) if named_loss else [loss for _, loss in loss_list]

  def _create_metric(self):
    metric_config = self._model_config.metric_config
    metric_list = []
    named_metric = False
    for metric_spec in metric_config.metric_spec:
      metric = None
      if metric_spec.HasField("standard_metric"):
        configurable_model_pb2.LossType.Name(
            metric_spec.standard_metric)[len("METRIC_TYPE_"):].lower()
      elif metric_spec.HasField("custom_metric"):
        metric = GVR.retrieve_callable(
            metric_spec.custom_metric.metric_function_registry_key)
      if metric is not None:
        metric_list.append((metric_spec.name, metric))
      named_metric = (named_metric and metric_spec.name)

    return dict(metric_list) if named_metric else [
        metric for _, metric in metric_list
    ]

  def _create_optimizer(self):
    if self._model_config.HasField("sgd_optimizer"):
      return _create_sgd_optimizer(self._model_config.sgd_optimizer)
    elif self._model_config.HasField("rmsprop_optimizer"):
      return _create_rmsprop_optimizer(self._model_config.rmsprop_optimizer)
    elif self._model_config.HasField("adagrad_optimizer"):
      return _create_adagrad_optimizer(self._model_config.adagrad.optimizers)
    elif self._model_config.HasField("adadelta_optimizer"):
      return _create_adadelta_optimizer(self._model_config.adadelta_optimizer)
    elif self._model_config.HasField("adam_optimizer"):
      return _create_adam_optimizer(self._model_config.adam_optimizer)
    else:
      return None

  def _create_layers(self):
    for layer_config in self._model_config.layer:
      self._evaluate_layer(layer_config)

  def init_from_config(self, model_config):
    self._model_config = model_config
    self._name_to_layer_config = dict(
        [l.name, l] for l in self._model_config.layer)
    self._create_layers()
    self._model = tf.keras.models.Model(name=self._model_config.name,
                                        inputs=self._inputs,
                                        outputs=self._outputs)
    optimizer = self._create_optimizer()
    assert optimizer, "Optimizer is not set."
    losses = self._create_loss()

    assert losses, "Loss is not set."
    metrics = self._create_metric()
    # Every metric must be a valid metric string or a function.
    assert all(metrics), "Invalid metric value."
    self._model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

  def _evaluate_layer(self, layer_config):
    if layer_config.name in self._tensor_lookup:
      return self._tensor_lookup[layer_config.name]

    dependent_tensors = [
        self._evaluate_layer(self._name_to_layer_config[d])
        for d in layer_config.dependency
    ]
    layer = None
    if layer_config.HasField("input"):
      layer = self._create_input_layer(layer_config.name, layer_config.input)
      self._inputs.append(layer)
    elif layer_config.HasField("flatten"):
      layer = self._create_flatten_layer(layer_config.name,
                                         layer_config.flatten)
    elif layer_config.HasField("dropout"):
      layer = self._create_dropout_layer(layer_config.name,
                                         layer_config.dropout)
    elif layer_config.HasField("embedding"):
      layer = self._create_embedding_layer(layer_config.name,
                                           layer_config.embedding)
    elif layer_config.HasField("conv_2d"):
      layer = self._create_conv_2d_layer(layer_config.name,
                                         layer_config.conv_2d)
    elif layer_config.HasField("max_pooling_2d"):
      layer = self._create_max_pooling_2d_layer(layer_config.name,
                                                layer_config.max_pooling_2d)
    elif layer_config.HasField("global_average_pooling_1d"):
      layer = self._create_global_average_pooling_1d(
          layer_config.name, layer_config.global_average_pooling_1d)
    elif layer_config.HasField("activation"):
      layer = self._create_activation_layer(layer_config.name,
                                            layer_config.activation)
    elif layer_config.HasField("text_vectorization"):
      layer = self._create_text_vectorization_layer(
          layer_config.name, layer_config.text_vectorization)
    elif layer_config.HasField("zero_padding_2d"):
      layer = self._create_zero_padding_2d_layer(layer_config.name,
                                                 layer_config.zero_padding)
    elif layer_config.HasField("l2_normalization"):
      layer = self._create_l2_normalization_layer(layer_config.name,
                                                  self.l2_normalization)
    elif layer_config.HasField("dense"):
      layer = self._create_dense_layer(layer_config.name, layer_config.dense)
    if layer is not None:
      tensor = layer(*dependent_tensors) if dependent_tensors else layer
      self._tensor_lookup[layer_config.name] = tensor
      if layer_config.is_output:
        self._outputs.append(tensor)
      return layer
    else:
      raise ValueError("invalid LayerConfig with missing layer specification.")

  def _create_dense_layer(self, name, layer):
    return tf.keras.layers.Dense(name=name,
                                 units=layer.units,
                                 activation=_activation_type_to_string(
                                     layer.activation),
                                 use_bias=layer.use_bias)

  def _create_dropout_layer(
      self, name: str,
      layer: configurable_model_pb2.DropoutLayer) -> tf.keras.layers.Dropout:
    return tf.keras.layers.Dropout(
        name=name,
        rate=layer.rate,
        noise_shape=(layer.noise_shape if layer.noise_shape else None),
        seed=(layer.seed if layer.seed else None))

  def _create_embedding_layer(
      self, name: str, layer: configurable_model_pb2.EmbeddingLayer
  ) -> tf.keras.layers.Embedding:
    return tf.keras.layers.Embedding(
        name=name,
        input_dim=layer.input_dim,
        output_dim=layer.output_dim,
        input_length=(layer.input_length if layer.input_length else None))

  def _create_flatten_layer(
      self, name: str,
      layer: configurable_model_pb2.FlattenLayer) -> tf.keras.layers.Flatten:
    del layer
    return tf.keras.layers.Flatten(name=name)

  def _create_input_layer(self, name, layer):
    return tf.keras.layers.Input(
        name=name,
        shape=tuple(layer.shape),
        batch_size=layer.batch_size if layer.batch_size else None,
        dtype=_non_empty_string_or_none(layer.dtype),
        sparse=layer.sparse)

  def _create_conv_2d_layer(self, name, layer):
    return tf.keras.layers.Conv2D(
        name=name,
        filters=layer.filters,
        kernel_size=(tuple(layer.kernel_size)
                     if len(layer.kernel_size) > 1 else layer.kernel_size[0]),
        strides=tuple(layer.strides),
        padding=_padding_type_to_string(layer.padding),
        data_format=_data_format_to_string(layer.data_format),
        use_bias=layer.use_bias,
        activation=_activation_type_to_string(layer.activation))

  def _create_global_average_pooling_1d(self, name, layer):
    return tf.keras.layers.GlobalAveragePooling1D()

  def _create_max_pooling_2d_layer(self, name, layer):
    return tf.keras.layers.MaxPooling2D(
        name=name,
        pool_size=(tuple(layer.pool_size)
                   if len(layer.pool_size) > 1 else layer.pool_size[0]),
        strides=(tuple(layer.strides)
                 if len(layer.strides) > 1 else layer.strides[0]),
        padding=_padding_type_to_string(layer.padding),
        data_format=_data_format_to_string(layer.data_format))

  def _create_activation_layer(self, name, layer):
    return tf.keras.layers.Activation(name=name,
                                      activation=_activation_type_to_string(
                                          layer.type))

  def _create_text_vectorization_layer(self, name, layer):
    standardize = 'lower_and_strip_punctuation'
    if layer.HasField('predefined_standardize'):
      standardize = layer.predefined_standardize
    else:
      standardize = GVR.retrieve_callable(layer.customized_standardize)

    return tf.keras.layers.TextVectorization(
        max_tokens=(layer.max_tokens if layer.max_tokens else None),
        standardize=standardize,
        split=(layer.split if layer.split else 'whitespace'),
        output_mode=(layer.output_mode if layer.output_mode else 'int'),
        vocabulary=(layer.vocab[:] if layer.vocab else None))

  def _create_zero_padding_2d_layer(self, name, layer):
    return tf.keras.layers.ZeroPadding2D(name=name,
                                         padding=tuple(layer.padding),
                                         data_format=_data_format_to_string(
                                             layer.data_format))

  def _create_l2_normalization_layer(self, name, layer):
    raise UnimplementedError("not implemented")