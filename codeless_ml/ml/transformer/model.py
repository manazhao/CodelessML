import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
  return pos / np.power(1000, (2 * i // 2) / np.float32(d_model))


def positional_encoding(position, d_model):
  angle_rads = get_angles(
      np.arange(position)[:, np.newaxis],
      np.arange(d_model)[np.newaxis, :], d_model)
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  # Expand the dimension so that the encoding can be applied to the examples in
  # the same batch.
  pos_encoding = angle_rads[np.newaxis, ...]  # (1, #position, d_model)
  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
  """ Create padding mask for the given sequence data.
   
  Create mask for the input token sequences. If a token is padding value, its
  mask value is set to 1 indicating the presence of padding.

  Arguments:
    seq: token sequences with shape (batch_size, seq_len).

  Returns:
    Mask tensor with shape (batch_size, 1, seq_len).

  Raises:
    ValueError: if `seq` rank is not 2.
  """
  # seq: (batch_size, seq_len)
  if tf.rank(seq) != 2:
    raise ValueError("input shape must be (batch_size, seq_len)")

  # 1 means the presence of padding value, 0 otherwise.
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # return shape (batch_size, 1, seq_len)
  return seq[:, tf.newaxis, :]


def create_look_ahead_mask(size):
  # 1 means the value shouldn't be used.
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  # Add a batch dimension.
  return mask[tf.newaxis, :, :]


def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (..., len_q, len_k)

  # apply the mask: setting the logits to negative infinity if the mask value is
  # 1 (padding).
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits,
                                    axis=-1)  # (..., seq_len_q, seq_len_k)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads):
    if d_model % num_heads:
      raise ValueError("d_model must be multiples of num_heads")

    super(MultiHeadAttention, self).__init__()
    self._d_model = d_model
    self._num_heads = num_heads

    self._depth = self._d_model // self._num_heads

    self._wq = tf.keras.layers.Dense(self._d_model)
    self._wk = tf.keras.layers.Dense(self._d_model)
    self._wv = tf.keras.layers.Dense(self._d_model)

    self._dense = tf.keras.layers.Dense(self._d_model)

  def _split_heads(self, x):
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    x = tf.reshape(
        x, shape=(batch_size, seq_len, self._num_heads,
                  self._depth))  # (batch_size, seq_len, num_heads, depth)
    return tf.transpose(x, perm=[0, 2, 1,
                                 3])  # (batch_size, num_heads, seq_len, depth)

  def call(self, q, k, v, mask):
    if tf.rank(q) != 3:
      raise ValueError(
          "q should be of shape [batch_size, seq_len, embedding_dim]")

    batch_size = tf.shape(q)[0]
    seq_len = tf.shape(q)[1]
    key_seq_len = tf.shape(k)[1]

    q = self._wq(q)  # (batch_size, seq_len, d_model)
    k = self._wk(k)  # (batch_size, key_seq_len, d_model)
    v = self._wv(v)  # (batch_size, key_seq_len, d_model)

    tf.ensure_shape(q, [batch_size, seq_len, self._d_model])
    tf.ensure_shape(k, [batch_size, key_seq_len, self._d_model])
    tf.ensure_shape(v, [batch_size, key_seq_len, self._d_model])

    q = self._split_heads(q)  # (batch_size, num_heads, seq_len, depth)
    k = self._split_heads(k)  # (batch_size, num_heads, key_seq_len, depth)
    v = self._split_heads(v)  # (batch_size, num_heads, key_seq_len, depth)
    tf.ensure_shape(q, [batch_size, self._num_heads, seq_len, self._depth])
    tf.ensure_shape(k, [batch_size, self._num_heads, key_seq_len, self._depth])
    tf.ensure_shape(v, [batch_size, self._num_heads, key_seq_len, self._depth])

    if mask is not None:
      tf.ensure_shape(mask, [batch_size, None, key_seq_len])
      # add a dimension for multi-heads.
      mask = mask[:, tf.newaxis, :, :]

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q=q, k=k, v=v, mask=mask)  # (batch_size, num_heads, seq_len, depth)
    tf.ensure_shape(scaled_attention,
                    [batch_size, self._num_heads, seq_len, self._depth])
    scaled_attention = tf.transpose(
        scaled_attention, perm=[0, 2, 1,
                                3])  # (batch_size, seq_len, num_heads, depth)
    concat_attention = tf.reshape(
        scaled_attention,
        shape=(batch_size, -1, self._d_model))  # (batch_size, seq_len, d_model)
    output = self._dense(concat_attention)  # (batch_size, seq_len, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(dff, d_model):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self._d_model = d_model
    self._mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    self._ffn = point_wise_feed_forward_network(dff=dff, d_model=d_model)

    self._dropout1 = tf.keras.layers.Dropout(rate)
    self._dropout2 = tf.keras.layers.Dropout(rate)

    self._layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, x, training, mask):
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    attn_output, _ = self._mha(q=x, k=x, v=x, mask=mask)
    tf.ensure_shape(attn_output, [batch_size, seq_len, self._d_model])
    # dropout the attention output.
    attn_output = self._dropout1(attn_output, training=training)
    # add mha input and mha output and normalize the output.
    out1 = self._layernorm1(x + attn_output)

    ffn_output = self._ffn(out1)
    ffn_output = self._dropout2(ffn_output)
    out2 = self._layernorm2(out1 + ffn_output)

    return out2


class Encoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               vocab_size,
               max_position_encoding,
               rate=0.1):
    super(Encoder, self).__init__()

    self._d_model = d_model
    self._num_layers = num_layers
    self._max_position_encoding = max_position_encoding

    self._embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self._pos_encoding = positional_encoding(max_position_encoding,
                                             self._d_model)

    self._enc_layers = [
        EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
    ]

    self._dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]

    x = self._embedding(x)  # (batch_size, seq_len, d_model)
    tf.ensure_shape(x, [batch_size, seq_len, self._d_model])

    x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
    # self._pos_encoding shape: (1, max_position_encoding, d_model)
    tf.ensure_shape(self._pos_encoding,
                    [1, self._max_position_encoding, self._d_model])
    x += self._pos_encoding[:, :seq_len, :]

    x = self._dropout(x, training=training)

    for i in range(self._num_layers):
      x = self._enc_layers[i](x, training, mask)

    tf.ensure_shape(x, [batch_size, seq_len, self._d_model])
    return x


class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self._mha1 = MultiHeadAttention(d_model, num_heads)
    self._mha2 = MultiHeadAttention(d_model, num_heads)

    self._ffn = point_wise_feed_forward_network(dff, d_model)

    self._layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self._layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self._dropout1 = tf.keras.layers.Dropout(rate)
    self._dropout2 = tf.keras.layers.Dropout(rate)
    self._dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    attn1, attn_weights_block1 = self._mha1(q=x, k=x, v=x, mask=look_ahead_mask)
    attn1 = self._dropout1(attn1, training=training)
    out1 = self._layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self._mha2(q=out1,
                                            k=enc_output,
                                            v=enc_output,
                                            mask=padding_mask)
    attn2 = self._dropout2(attn2, training=training)
    out2 = self._layernorm2(out1 + attn2)

    ffn_output = self._ffn(out2)
    ffn_output = self._dropout3(ffn_output, training=training)
    out3 = self._layernorm3(ffn_output)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               vocab_size,
               max_position_encoding,
               rate=0.1):
    super(Decoder, self).__init__()

    self._d_model = d_model
    self._num_layers = num_layers
    self._max_position_encoding = max_position_encoding

    self._embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    self._pos_encoding = positional_encoding(max_position_encoding,
                                             self._d_model)

    self._dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
        for _ in range(num_layers)
    ]

    self._dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self._embedding(x)
    x *= tf.math.sqrt(tf.cast(self._d_model, tf.float32))
    x += self._pos_encoding[:, :seq_len, :]

    x = self._dropout(x, training=training)

    for i in range(self._num_layers):
      x, block1, block2 = self._dec_layers[i](x=x,
                                              enc_output=enc_output,
                                              training=training,
                                              look_ahead_mask=look_ahead_mask,
                                              padding_mask=padding_mask)
      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    return x, attention_weights


class Transformer(tf.keras.Model):

  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               target_vocab_size,
               max_pe_input,
               max_pe_target,
               rate=0.1):
    super(Transformer, self).__init__()

    self._d_model = d_model
    self._target_vocab_size = target_vocab_size

    self._encoder = Encoder(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            vocab_size=input_vocab_size,
                            max_position_encoding=max_pe_input,
                            rate=rate)
    self._decoder = Decoder(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            vocab_size=target_vocab_size,
                            max_position_encoding=max_pe_target,
                            rate=rate)
    self._final_layer = tf.keras.layers.Dense(target_vocab_size)

  def _create_masks(self, inp, tar):
    batch_size = tf.shape(inp)[0]
    seq_len = tf.shape(inp)[1]
    # Encoder padding mask.
    enc_padding_mask = create_padding_mask(inp)
    tf.ensure_shape(enc_padding_mask, [batch_size, 1, seq_len])

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    enc_output_mask = create_padding_mask(inp)
    tf.ensure_shape(enc_output_mask, [batch_size, 1, seq_len])

    # Used in the first attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the
    # decoder.
    target_seq_len = tf.shape(tar)[1]
    look_ahead_mask = create_look_ahead_mask(target_seq_len)
    tf.ensure_shape(look_ahead_mask, [1, target_seq_len, target_seq_len])
    target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.maximum(target_padding_mask, look_ahead_mask)
    tf.ensure_shape(look_ahead_mask,
                    [batch_size, target_seq_len, target_seq_len])

    return enc_padding_mask, look_ahead_mask, enc_output_mask

  def call(self, inputs, training):
    inp, tar = inputs
    batch_size = tf.shape(inp)[0]
    target_seq_len = tf.shape(tar)[1]

    enc_padding_mask, look_ahead_mask, enc_output_mask = self._create_masks(
        inp, tar)

    enc_output = self._encoder(x=inp, training=training, mask=enc_padding_mask)

    dec_output, attention_weights = self._decoder(
        x=tar,
        enc_output=enc_output,
        training=training,
        look_ahead_mask=look_ahead_mask,
        padding_mask=enc_output_mask)
    tf.ensure_shape(dec_output, [batch_size, target_seq_len, self._d_model])

    final_output = self._final_layer(
        dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    tf.ensure_shape(final_output,
                    [batch_size, target_seq_len, self._target_vocab_size])

    return final_output, attention_weights
