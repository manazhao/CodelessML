import os

import tensorflow_datasets as tfds

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_name", None,
                    "a valid dataset name from `tfds.list_builders()`")
flags.DEFINE_enum("split", None, ["train", "test", "valid", "validation"],
                  "dataset split for train, validation and evaluation.")
flags.DEFINE_string("data_dir", None,
                    "directory holding downloaded tfds datasets.")

flags.mark_flag_as_required("dataset_name")
flags.mark_flag_as_required("split")


def main(argv):
    dataset_name = FLAGS.dataset_name
    # assert dataset_name in tfds.list_builders()

    data_dir = FLAGS.data_dir
    if data_dir is None:
        logging.info(
            "data_dir flag is not set, try to use the value from the TFDS_DATA_DIR environment variable"
        )
        data_dir = os.environ.get("TFDS_DATA_DIR")

    if data_dir is None:
        logging.fatal(
            "the data_dir must be set by flag or by the TFDS_DATA_DIR env variable"
        )

    dataset, info = tfds.load(dataset_name,
                              split=FLAGS.split,
                              data_dir=data_dir,
                              batch_size=10,
                              download=True,
                              with_info=True)
    print(f"dataset info: {info}")
    for batch in dataset:
        print(f"sample {sample_cnt}: {batch}")
        break


if __name__ == "__main__":
    app.run(main)
