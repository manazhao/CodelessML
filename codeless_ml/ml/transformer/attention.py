import tensorflow as tf

from codeless_ml.ml.transformer.cache_attention import CacheAttention


class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = CacheAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):

    def call(self, x, context, reset_cache: tf.Tensor | None = None):
        attn_output, attn_scores = self.mha(query=x,
                                            key=context,
                                            value=context,
                                            reset_cache=reset_cache,
                                            return_attention_scores=True)
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):

    def call(self, x, reset_cache: tf.Tensor | None = None):
        attn_output = self.mha(query=x,
                               value=x,
                               key=x,
                               reset_cache=reset_cache)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):

    def call(self, x, reset_cache: tf.Tensor | None = None):
        attn_output = self.mha(query=x,
                               value=x,
                               key=x,
                               use_causal_mask=True,
                               reset_cache=reset_cache)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
