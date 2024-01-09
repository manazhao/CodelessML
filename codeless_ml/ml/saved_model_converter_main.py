from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to a checkpoint file.")
flags.DEFINE_string("saved_model_path", None,
                    "Path to the resulted SavedModel.")
flags.DEFINE_bool("serving_only", True,
                  "Whether the SavedModel will be used only for serving.")

flags.mark_flag_as_required("checkpoint_path")


def main(argv):
  del argv
  tf.enable_eager_execution()
  model = tf.keras.models.load_model(filepath=FLAGS.checkpoint_path)
  model.summary()
  logging.info("SavedModel path: %s" % (FLAGS.saved_model_path))
  tf.contrib.saved_model.save_keras_model(
      model, FLAGS.saved_model_path, serving_only=FLAGS.serving_only)


if __name__ == "__main__":
  app.run(main)
