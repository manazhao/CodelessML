import unittest

from absl import logging

import google.protobuf.text_format as text_format
import tensorflow_datasets as tfds
import tensorflow_text

import codeless_ml.ml.input_pb2 as input_pb2
import codeless_ml.ml.input as input
import codeless_ml.ml.registry.mtnt as mtnt

_BATCH_SIZE = 5


class TranslationTest(unittest.TestCase):

    def test_load_and_tokenize(self):
        dataset_config_pbtxt = f"""
            tfds {{
                # only load 10 rows of data.
                name: "ted_hrlr_translate/pt_to_en"
                split: "train[:100]"
                data_dir: "codeless_ml/ml/artifacts"
            }}
            batch_size: {_BATCH_SIZE}
            repeat: 1

            map_callable {{
                function_name: "{mtnt.TRANSLATE_PREPARE_BATCH_REGISTRY_NAME}"
            }}
        """
        dataset_config = input_pb2.DatasetConfig()
        text_format.Parse(dataset_config_pbtxt, dataset_config)
        data = input.get_dataset(dataset_config)
        for ele in data:
            pt_en_tensors, label = ele
            en_tensors = pt_en_tensors["en"]
            pt_tensors = pt_en_tensors["pt"]
            self.assertEqual(en_tensors.shape, label.shape)
            self.assertEqual(pt_tensors.shape[0], en_tensors.shape[0])


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    unittest.main()
