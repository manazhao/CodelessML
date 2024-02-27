import numpy.testing as npt
import tensorflow as tf

import codeless_ml.ml.registry.vae as vae
import codeless_ml.common.callable_pb2 as callable_pb2
import codeless_ml.common.global_variable as gv


class VaeTest(tf.test.TestCase):

    def test_create_layer(self):
        registry = callable_pb2.CallableRegistry()
        registry.closure.function_name = vae.CREATE_LAYER_FUNCTION_NAME
        layer = gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(registry)
        batch_size, latent_dim = 2, 5
        x = tf.random.normal([batch_size, latent_dim * 2])
        z = layer(x)
        self.assertEqual(z.shape, [batch_size, latent_dim])

    def test_binary_image(self):
        up_half_image = tf.constant(10, shape=[14, 28, 1], dtype=tf.float32)
        down_half_image = tf.constant(200, shape=[14, 28, 1], dtype=tf.float32)
        image = tf.concat([up_half_image, down_half_image], axis=0)
        registry = callable_pb2.CallableRegistry()
        registry.function_name = vae.INPUT_PROCESS_FUNCTION_NAME
        process_fn = gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(registry)
        binary_image, _ = process_fn(image, label=None)
        expected_image = tf.concat([
            tf.constant(0., shape=[14, 28, 1], dtype=tf.float32),
            tf.constant(1., shape=[14, 28, 1], dtype=tf.float32)
        ],
                                   axis=0)
        npt.assert_allclose(binary_image, expected_image)


if __name__ == "__main__":
    tf.test.main()
