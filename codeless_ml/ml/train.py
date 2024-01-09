from absl import flags
from absl import app

from google.protobuf import text_format
from codeless_ml.ml.train_pb2 import ModelTrainerConfig
from codeless_ml.ml.model_trainer import ModelTrainer

FLAGS = flags.FLAGS

flags.DEFINE_string("trainer_config_file", None,
                    "Path to a ModelTrainerConfig protobuf text file")
flags.DEFINE_enum("job", None, ["train", "evaluate"],
                  "whether to train or evaluate the model.")

flags.mark_flag_as_required("trainer_config_file")
flags.mark_flag_as_required("job")


def main(argv):
  with open(FLAGS.trainer_config_file, "r") as f:
    trainer_config = ModelTrainerConfig()
    config_pbtxt = f.read()
    text_format.Parse(config_pbtxt, trainer_config)

  trainer = ModelTrainer()
  trainer.init_from_config(trainer_config)
  if FLAGS.job == "train":
    trainer.train()
  elif FLAGS.job == "evaluate":
    trainer.evalaute()
  else:
    logging.fatal("unsupported job: %s" % (FLAGS.job))


if __name__ == "__main__":
  app.run(main)
