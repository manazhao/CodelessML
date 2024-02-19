import tensorflow as tf

from codeless_ml.ml.transformer.encoder import Encoder
from codeless_ml.ml.transformer.decoder import Decoder


class Transformer(tf.keras.Model):

    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 input_vocab_size: int,
                 target_vocab_size: int,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)  # (batch_size, context_len, d_model)
        x = self.decoder(x, context)  # (batch_size, target_len, d_model)
        logits = self.final_layer(
            x)  # (batch_size, target_len, target_vocab_size)
        return logits
