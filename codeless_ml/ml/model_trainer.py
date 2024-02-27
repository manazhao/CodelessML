import datetime
import importlib
import os

from absl import logging
from typing import Callable, Any, TypeVar, List, Mapping
from google.protobuf import text_format

import tensorflow as tf
import tensorflow_text

from codeless_ml.ml.train_pb2 import ModelTrainerConfig
from codeless_ml.ml.input_pb2 import DatasetConfig
from codeless_ml.ml.train_pb2 import ModelCheckpointConfig
from codeless_ml.ml.configurable_model import ConfigurableModel
import codeless_ml.common.global_variable as gv
import codeless_ml.ml.input as data_input

GVR = gv.GLOBAL_VARIABLE_REPOSITORY


class ModelTrainer(object):

    def __init__(self):
        self._trainer_config = ModelTrainerConfig()
        self._configurable_model = ConfigurableModel()
        self._train_dataset = None
        self._validation_dataset = None
        self._evaluation_dataset = None

    def _init_model(self):
        # Order of initializing the model:
        # 1. Init the architecture from ConfigurableModel config if
        #    load_model_config is not given.
        # 2. If full_model_path is given, load both the architecture and weights
        #    from the file.
        # 3. Try to initialize the architecture from json_file, yaml_file and then
        #    the ModelConfig.
        # 4. Load the weights if weights_file is given.
        if not self._trainer_config.HasField("load_model_config"):
            self._configurable_model.init_from_config(
                self._trainer_config.model_config)
            return

        load_config = self._trainer_config.load_model_config
        if load_config.model_path:
            logging.info("Load full model path: %s" % (load_config.model_path))
            self._configurable_model.model = tf.keras.models.load_model(
                load_config.model_path)
            return

        if load_config.saved_model_path:
            self._configurable_model.model = \
                tf.keras.models.load_model(load_config.saved_model_path)
            return

        if load_config.architecture_path:
            with open(load_config.architecture_path) as f:
                json_str = f.read()
                self._configurable_model.model = tf.keras.models.model_from_json(
                    json_str)
        if load_config.weights_path:
            self._configurable_model.model.load_weights(
                load_config.weights_path)

    def init_from_config(self, trainer_config: ModelTrainerConfig) -> bool:
        logging.info("train_config: %s" % str(trainer_config))
        self._trainer_config = trainer_config
        # Load the python modules if they're configured.
        for module_name in self._trainer_config.user_defined_python_module:
            logging.info("Load python module %s" % (module_name))
            importlib.import_module(module_name)
        logging.info("create model from config.")
        self._init_model()
        self._configurable_model.model.summary()
        return True

    @property
    def configurable_model(self) -> ConfigurableModel:
        return self._configurable_model

    def _tensorboard_callback(self):
        tensor_board_config = self._trainer_config.tensor_board_config
        update_freq = tensor_board_config.WhichOneof("update_freq")
        if update_freq == "samples":
            update_freq = tensor_board_config.samples
        return tf.keras.callbacks.TensorBoard(
            log_dir=tensor_board_config.log_dir,
            batch_size=tensor_board_config.batch_size,
            write_graph=tensor_board_config.write_graph,
            write_images=tensor_board_config.write_images,
            write_grads=tensor_board_config.write_grads,
            update_freq=update_freq)

    def _custom_callbacks(self, model, validation_data):
        callbacks_config = self._trainer_config.custom_callback_config
        callbacks = []
        for registry in callbacks_config.registry:
            callback = GVR.retrieve_callable(registry)
            callback.set_model(model)
            callback.validation_data = validation_data
            callbacks.append(callback)
        return callbacks

    def _checkpoint_callback(self):
        checkpoint_config = self._trainer_config.checkpoint_config
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_config.filepath,
            monitor=(checkpoint_config.monitor
                     if checkpoint_config.monitor else "val_loss"),
            save_best_only=True,
            save_weights_only=False,
            mode=("auto" if checkpoint_config.mode
                  == ModelCheckpointConfig.SAVE_MODE_UNSPECIFIED else
                  checkpoint_config.mode),
            save_freq="epoch")

    def train(self) -> tf.keras.callbacks.History:
        logging.info("Create training dataset.")
        self._train_dataset = data_input.get_dataset(
            self._trainer_config.train_dataset)
        if self._trainer_config.HasField("validation_dataset"):
            logging.info("Create validation dataset.")
            self._validation_dataset = data_input.get_dataset(
                self._trainer_config.validation_dataset)
        logging.info("start training...")
        fit_config = self._trainer_config.fit_config
        callbacks = []
        if self._trainer_config.HasField("tensor_board_config"):
            callbacks.append(self._tensorboard_callback())
        if self._trainer_config.HasField("checkpoint_config"):
            callbacks.append(self._checkpoint_callback())
        if self._trainer_config.HasField("custom_callback_config"):
            callbacks.extend(
                self._custom_callbacks(self._configurable_model.model,
                                       self._validation_dataset))
        return self._configurable_model.model.fit(
            x=self._train_dataset,
            y=None,
            validation_data=self._validation_dataset,
            callbacks=callbacks,
            epochs=fit_config.epochs,
            steps_per_epoch=(fit_config.steps_per_epoch
                             if fit_config.steps_per_epoch else None),
            validation_steps=(fit_config.validation_steps
                              if fit_config.validation_steps else None))

    def _mkdir_if_not_exists(self, path):
        if os.path.exists(path):
            logging.info("Path already exists: %s" % (path))
            return
        try:
            original_umask = os.umask(0)
            os.makedirs(path, mode=0o755)
        finally:
            os.umask(original_umask)

    def save_model(self) -> str:
        if not self._trainer_config.HasField("save_model_config"):
            logging.warn("SaveModelConfig is not set. Model wont'be saved")
            return

        save_config = self._trainer_config.save_model_config
        epoch = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        full_model_file = os.path.join(save_config.output_directory,
                                       "full_model_%s.keras" % (epoch))
        logging.info("save full model: %s" % (full_model_file))
        self._configurable_model.model.save(full_model_file)
        return full_model_file

    def evaluate(self) -> TypeVar("EvalResult", float, List[float]):
        logging.info("Create evaluation dataset.")
        assert self._trainer_config.HasField("evaluation_dataset"), \
            "evaluation dataset must be specified."
        self._evaluation_dataset = data_input.get_dataset(
            self._trainer_config.evaluation_dataset)
        logging.info("start evaluation...")
        return self._configurable_model.model.evaluate(
            self._evaluation_dataset,
            steps=self._trainer_config.evaluate_config.steps)
