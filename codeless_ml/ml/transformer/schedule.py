import tensorflow as tf

from keras.utils import register_keras_serializable

import codeless_ml.common.global_variable as gv

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


@register_keras_serializable(package="codeless_ml.ml.transformer")
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(tf.cast(
            self.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)


def _create_custom_schedule(d_model: int, warmup_steps: int = 4000):
    return CustomSchedule(d_model, warmup_steps)


CREATE_SCHEDULE_REGESTRY_KEY = "/optimizer/transformer/custom_schedule"

GVR.register_callable(CREATE_SCHEDULE_REGESTRY_KEY, _create_custom_schedule)
