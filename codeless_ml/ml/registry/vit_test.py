import unittest
import numpy as np

import google.protobuf.text_format as text_format
import tensorflow as tf

import codeless_ml.ml.registry.vit
import codeless_ml.common.callable_pb2 as callable_pb2
import codeless_ml.common.global_variable as gv


class VitTest(unittest.TestCase):

    def test_concat(self):
        cls_embedding = tf.constant([[[1.0, 2.0]], [[3.0, 4.0]]])
        patch_embedding = tf.constant([[[5.0, 6.0], [7.0, 8.0]],
                                       [[9.0, 10.0], [11.0, 12.0]]])
        callable_registry = callable_pb2.CallableRegistry()
        callable_registry.function_name = "codeless_ml.ml.registry.concat_embeddings"
        tf_func = gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(
            callable_registry)
        np.testing.assert_allclose(tf_func(cls_embedding, patch_embedding),
                                   [[[1.0, 2.0], [5.0, 6.0], [7.0, 8.0]],
                                    [[3.0, 4.0], [9.0, 10.0], [11.0, 12.0]]])

    def test_extract_cls_embedding(self):
        emb = tf.constant([[[.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                           [[.0, 1.0], [2.0, 3.0], [4.0, 5.0]]])
        callable_registry = callable_pb2.CallableRegistry()
        callable_registry.function_name = "codeless_ml.ml.registry.extract_cls_embedding"
        tf_func = gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(
            callable_registry)
        np.testing.assert_allclose(tf_func(emb), [[.0, 1.0], [.0, 1.0]])

    def _create_image(self, val, w, h, c):
        return tf.reshape(tf.range(w * h * c) + val, [h, w, c])

    def test_input_fn(self):
        w, h, c = 10, 10, 3
        image = self._create_image(1, w, h, c)
        callable_registry_pbtxt = f"""
            closure {{
                function_name: "codeless_ml.ml.registry.create_input_fn"
                argument {{
                    key: "new_height"
                    value {{ int32_value: {h} }}
                }}
                argument {{
                    key: "new_width"
                    value {{ int32_value: {w} }}
                }}
                argument {{
                    key: "num_patches"
                    value {{ int32_value: 2 }}
                }}
            }}
        """
        callable_registry = callable_pb2.CallableRegistry()
        text_format.Parse(callable_registry_pbtxt, callable_registry)
        input_fn = gv.GLOBAL_VARIABLE_REPOSITORY.retrieve_callable(
            callable_registry)
        label = tf.constant([1])
        new_features, y = input_fn(image, label)
        np.testing.assert_allclose(y, label)
        patch, pos_token, cls_token = (new_features["patch"],
                                       new_features["pos_token"],
                                       new_features["cls_token"])
        self.assertEqual(patch.shape, (4, 5 * 5 * 3))
        np.testing.assert_allclose(pos_token, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(cls_token, [0])


if __name__ == '__main__':
    unittest.main()
