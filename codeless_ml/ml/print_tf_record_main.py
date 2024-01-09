from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("tf_record_file", None, "TfRecord file.")
flags.DEFINE_integer("num_samples", 10, "Number of samples to print.")


def main(argv):
  del argv
  record_iterator = tf.io.tf_record_iterator(path=FLAGS.tf_record_file)
  count = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    print("example: %d" % (count))
    print(example)
    count += 1
    if count >= FLAGS.num_samples:
      break


if __name__ == "__main__":
  app.run(main)
