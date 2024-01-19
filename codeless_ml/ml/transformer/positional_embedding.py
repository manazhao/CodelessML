import numpy as np
import tensorflow as tf

from keras.utils import register_keras_serializable


@register_keras_serializable(package="codeless_ml.ml.transformer")
class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 max_length: int = 2048,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   d_model,
                                                   mask_zero=True)
        self.positional_embedding = self._positional_embedding()

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def _positional_embedding(self):
        positions = np.arange(self.max_length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(
            self.d_model)[np.newaxis, :] / self.d_model  # (1, d_model)
        angle_rates = 1 / (10000**depths)  # (1, depth)
        pos_encoding = positions * angle_rates  # (pos, length)
        # odd positions use cos and even positions use sin
        odd_cols = np.arange(1, self.d_model, 2)
        pos_encoding[:, odd_cols] = np.cos(pos_encoding[:, odd_cols])
        even_cols = np.arange(0, self.d_model, 2)
        pos_encoding[:, even_cols] = np.sin(pos_encoding[:, even_cols])
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]  # x shape: [B, L]
        x = self.embedding(x)  # (B, L, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.positional_embedding[tf.newaxis, :length, :]
        return x

    def get_config(self):
        config = super().get_config().copy()
        # note: don't write positional_embedding member to the config.
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_length": self.max_length
        })
        return config
