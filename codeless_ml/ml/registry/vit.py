import tensorflow as tf

import codeless_ml.common.global_variable as gv


def concat_embeddings(cls_embedding, patch_embedding):
    # cls_embedding's shape: (B, 1, D)
    # patch_embedding's shape: (B, L, D)
    # concat the embedding tensors on the second dimension.
    return tf.concat([cls_embedding, patch_embedding], axis=1)


def extract_cls_embedding(emb):
    # embedding for the [CLS] token should be the first one.
    # emb shape: [B, L, D]
    # return value shape: [B, D]
    return tf.squeeze(emb[:, 0:1, :], axis=1)


def create_input_fn(new_height: int, new_width: int, num_patches: int):
    assert new_height % num_patches == 0, "image height must be multiples of the patch height"
    assert new_width % num_patches == 0, "image height must be multiples of the patch width"

    def _fn(image, label):
        # images shape: (H, W, 3)
        image = tf.image.resize_with_pad(image,
                                         target_height=new_height,
                                         target_width=new_width)
        # now divide the image into patches.
        patch_h = new_height // num_patches
        patch_w = new_width // num_patches
        patches = []
        for r in range(num_patches):
            row_from = r * patch_h
            row_to = row_from + patch_h
            for c in range(num_patches):
                col_from = c * patch_w
                col_to = col_from + patch_w
                patch = image[row_from:row_to, col_from:col_to, :]
                # flatten the patch.
                patch = tf.reshape(patch, [1, -1])  # shape: [1, L]
                patches.append(patch)
        # the return tensor shape: (N, L] where N=num_patches * num_patches.
        patch_tensor = tf.concat(patches, axis=0)
        # additional token for [CLS].
        pos_tokens = tf.range(num_patches * num_patches + 1)
        cls_tokens = [0]
        return {
            "pos_token": pos_tokens,
            "patch": patch_tensor,
            "cls_token": cls_tokens
        }, label

    return _fn


gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(
    "codeless_ml.ml.registry.concat_embeddings", concat_embeddings)
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(
    "codeless_ml.ml.registry.extract_cls_embedding", extract_cls_embedding)
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(
    "codeless_ml.ml.registry.create_input_fn", create_input_fn)
