import tensorflow as tf

import codeless_ml.ml.configurable_model_pb2 as cm_pb2
import codeless_ml.ml.configurable_model as cm

from typing import Dict


class AutoRegressiveInference(tf.Module):

    def __init__(self,
                 config: cm_pb2.ModelConfig,
                 decoder_end_token: int,
                 cache_max_seq_len: int = 128):
        # modify the config by setting the `cache_max_seq_len` of the decoder
        # layer to `cache_max_seq_len`.
        self.decoder_end_token = decoder_end_token
        self.input_names = set()
        self.decoder_input_name = None
        for layer in self.confi.layer:
            if layer.HasField('input'):
                self.input_names.add(layer.name)

        for layer in self.config.layer:
            if not layer.HasField('transformer_decoder'):
                continue

            config.transformer_decoder.cache_max_seq_len = \
                     self.cache_max_seq_len
            self.decoder_input_name = [
                d.name for d in layer.dependency()
                if d.name in self.input_names
            ]
            break

        assert len(self.decoder_input_name) == 1, \
                "only one input is expected for the decoder"
        configurable_model = cm.ConfigurableModel()
        configurable_model.init_from_config(self.config)
        self.model = configurable_model.model

    def __call__(self, inputs: Dict[str, tf.Tensor]):
        assert isinstance(inputs, dict), "inputs must be a dict of tensors"
        # ensure the tensors in the `inputs` are the same as self.input_name.
        assert set(inputs.keys()) == self.input_names, \
                "the inputs must be the same inputs used in model config"

        # the decoder input must already hold one element, i.e., the [START]
        # token.
        tf.ensure_shape(inputs[self.decoder_input_name].shape, [None, 1])
        output_array = tf.TensorArray(dtype=tf.int64,
                                      size=0,
                                      dynamic_size=True)
        decoder_input = inputs[self.decoder_input_name]
        for i in range(self.cache_max_seq_len):
            predictions = self.model(inputs, training=False)
            # TODO: get the most likely token/index of the `predictions and feed
            # it into the decoder_input to predict the next token. since we are
            # using cached attention, decoder_input only needs to contain the
            # current token.
