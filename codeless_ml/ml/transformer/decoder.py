import tensorflow as tf

from typing import Union

from keras.utils import register_keras_serializable

from codeless_ml.ml.transformer.attention import CausalSelfAttention, CrossAttention
from codeless_ml.ml.transformer.ff import FeedForward
from codeless_ml.ml.transformer.positional_embedding import PositionalEmbedding


@register_keras_serializable(package="codeless_ml.ml.transformer")
class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 dropout_rate=0.1,
                 cache_max_seq_len=None,
                 **kwargs):
        """
      	Initializes a DecoderLayer instance, configuring its internal components.

      	Args:
      	    d_model (int): Dimensionality of the model's internal
                representations.
      	    num_heads (int):  Number of attention heads for multi-head
                attention.
      	    dff (int): Dimensionality of the feed-forward network.
      	    dropout_rate (float): Dropout rate (used for regularization).
      	    cache_max_seq_len (int): When set, cache will be enabled for the
                MHA layer and the multi-head key value tensors will be cached
                in preallocated memory. Note: this should only be set for model
                inference.
      	    **kwargs: Additional keyword arguments.
      	"""
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.cache_max_seq_len = cache_max_seq_len

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            cache_max_seq_len=cache_max_seq_len)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            cache_max_seq_len=cache_max_seq_len)

        self.ffn = FeedForward(d_model=d_model, dff=dff)

    def call(self, x, context, reset_cache: tf.Tensor | None = None):
        x = self.causal_self_attention(x=x, reset_cache=reset_cache)
        x = self.cross_attention(x=x, context=context, reset_cache=reset_cache)
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape (batch_size, seq_len, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "cache_max_seq_len": self.cache_max_seq_len
        })
        return config


@register_keras_serializable(package="codeless_ml.ml.transformer")
class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 vocab_size: Union[int, None],
                 dropout_rate: float = 0.1,
                 cache_max_seq_len=None,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.cache_max_seq_len = cache_max_seq_len

        self.pos_embedding = None
        if self.vocab_size is not None:
            self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                     d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate,
                         cache_max_seq_len=cache_max_seq_len)
            for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context, reset_cache: tf.Tensor | None = None):
        # apply the positional embedding if input is expected to be tokens.
        if self.pos_embedding is not None:
            x = self.pos_embedding(x)  # (batch_size, seq_len, d_model)
        else:
            tf.ensure_shape(x, (None, None, self.d_model))
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context, reset_cache=reset_cache)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        # x shape: (batch_size, target_seq_len, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate,
            "cache_max_seq_len": self.cache_max_seq_len
        })
        return config
