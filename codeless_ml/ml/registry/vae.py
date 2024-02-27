from absl import logging

import numpy as np
import tensorflow as tf

import codeless_ml.common.global_variable as gv

from keras.utils import register_keras_serializable


class LatentSample(tf.keras.layers.Layer):

    def call(self, inputs):
        # B: batch size
        # D; latent dim
        # shape: (B, D * 2)
        tf.ensure_shape(inputs, [None, None])
        # mean and logvar shape: [B, D]
        mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
        # shape: (B, D)
        z = self._reparam(mean, logvar)
        # compute the loss specific to this layer, ie.., log(p(zi)) -
        # log(q(zi|xi).
        logpz = self._log_normal_pdf(z, mean=.0, logvar=.0)
        logqz_x = self._log_normal_pdf(z, mean, logvar)
        self.add_loss(-tf.reduce_mean(logpz - logqz_x))
        return z

    def _reparam(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        z = mean + eps * tf.exp(logvar * 0.5)
        return z

    def _log_normal_pdf(self, z, mean, logvar):
        # z shape: (B, D)
        # log p(z|mu, var) = -log(sqrt(2*pi*var)) - (x-a)^2/(2*var)
        # var = logvar^2
        return tf.reduce_sum(-0.5 * (tf.math.log(2 * np.pi) + logvar +
                                     (z - mean)**2 * tf.exp(-logvar)),
                             axis=1)


def convert_to_binary_image(image, label):
    del label
    tf.ensure_shape(image, [28, 28, 1])
    image = tf.cast(image, dtype=tf.float32) / 255.0
    binary_image = tf.where(image > 0.5, 1.0, 0.0)
    return binary_image, binary_image


def create_layer():
    return LatentSample()


@register_keras_serializable(package="codeless_ml.ml.transformer")
def cross_entropy_loss(labels, logits):
    # compute cross entropy between original and reconstructed.
    tf.ensure_shape(labels, [None, None, None, 1])
    tf.ensure_shape(logits, [None, None, None, 1])
    ce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits),
                       axis=[1, 2, 3])
    return tf.reduce_mean(ce)


class PlotImagesCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir: str):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.log_dir = log_dir
        logging.info(f"create file writer under {self.log_dir}")
        self.file_writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # make predictions with the images in the validation set and display the
        # results on tensorboard.
        predictions = self.model.predict(self.validation_data, steps=1)
        with self.file_writer.as_default():
            tf.summary.image("validation samples",
                             predictions,
                             max_outputs=16,
                             step=epoch)


def create_plot_images_callback(log_dir: str):
    return PlotImagesCallback(log_dir)


INPUT_PROCESS_FUNCTION_NAME = 'codeless_ml.ml.registry.vae.convert_to_binary_image'
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(INPUT_PROCESS_FUNCTION_NAME,
                                                convert_to_binary_image)

CREATE_LAYER_FUNCTION_NAME = 'codeless_ml.ml.registry.vae.create_layer'
logging.info(
    f"Register function for creating LatentSample layer: {CREATE_LAYER_FUNCTION_NAME}"
)
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(CREATE_LAYER_FUNCTION_NAME,
                                                create_layer)

LOSS_FUNCTION_NAME = 'codeless_ml.ml.registry.vae.loss'
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(LOSS_FUNCTION_NAME,
                                                cross_entropy_loss)

CREATE_PLOT_IMAGES_CALLBACK_FUNCTION_NAME = \
        'codeless_ml.ml.registry.vae.create_plot_images_callback'
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(
    CREATE_PLOT_IMAGES_CALLBACK_FUNCTION_NAME, create_plot_images_callback)
