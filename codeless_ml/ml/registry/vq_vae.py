import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):

    def __init__(self,
                 latent_dim: int,
                 vocab_size: int,
                 beta: float = 0.2,
                 init_value: tf.Tensor | None = None):
        self._latent_dim = latent_dim
        self._vocab_size = vocab_size
        self._beta = beta
        if init_value is None:
            init_value = tf.random_normal_initializer()(shape=(vocab_size,
                                                               latent_dim),
                                                        dtype=tf.float32)

        w_init = tf.random_normal_initializer()
        self._embeddings = tf.Variable(
            initial_value=init_value,
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # shape: (w, h, depth)
        tf.ensure_shape(x, [None, None, self._latent_dim])
        input_shape = x.shape
        # shape: (w*h, d)
        flattened = tf.reshape(x, [-1, self._.latent_dim])
        codes = self._get_codes(flattened)
        # shape: (w*h, K)
        encodings = tf.one_hot(codes, self._vocab_size)
        # quantized input: (w*h, d)
        q_x = tf.matmul(encodings, self._embeddings)
        qx = tf.reshape(q_x, input_shape)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(q_x) - x)**2)
        codebook_loss = tf.reduce_mean((q_x - tf.stop_gradient(x))**2)
        self.add_loss(self._beta * commitment_loss + codebook_loss)
        # tricky bit: straight-through estimator which copies the gradient of qx
        # from the decoder to x.
        qx = x + tf.stop_gradient(qx - x)
        return qx

    def _get_codes(self, x):
        # compute the l2 distance of each row of `x` to the latent vectors in
        # the codebook and find the closest code.
        # l2 distance  = |(x_i - y_j||^2=x_i.x_i + y_j.y_j - 2.x_i.y_j where .
        # means dot product.
        # get x^2: shape: (M, 1)
        xx = tf.reduce_sum(x**2, axis=1, keepdims=True)
        # get y^2: shape: (K,)
        # note: reduce_sum will reduce the dims by 1.
        yy = tf.reduce_sum(self._embeddings**2, axis=1)
        # shape: (M, K)
        xy = x * self._embeddings
        # shape: (M, K)
        l2_dist = xx + yy - 2 * xy
        return tf.argmin(l2_dist, axis=1)
