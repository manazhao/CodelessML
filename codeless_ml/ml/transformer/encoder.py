import tensorflow as tf

from keras.utils import register_keras_serializable

from codeless_ml.ml.transformer.attention import GlobalSelfAttention, CrossAttention, CausalSelfAttention
from codeless_ml.ml.transformer.ff import FeedForward
from codeless_ml.ml.transformer.positional_embedding import PositionalEmbedding


@register_keras_serializable(package="codeless_ml.ml.transformer")
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = GlobalSelfAttention(num_heads=num_heads,
                                                  key_dim=d_model,
                                                  dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


@register_keras_serializable(package="codeless_ml.ml.transformer")
class Encoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 vocab_size: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)  # shape (batch_size, seq_len, d_model)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate
        })
        return config
