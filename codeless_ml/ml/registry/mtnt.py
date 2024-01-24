import tensorflow as tf

import codeless_ml.common.global_variable as gv


def _get_tokenizers():
    model_path = "codeless_ml/ml/artifacts/ted_hrlr_translate_pt_en_converter"
    tokenizers = tf.saved_model.load(model_path)
    return tokenizers.en, tokenizers.pt


MAX_TOKENS = 128
EN_TOKENIZER, PT_TOKENIZER = _get_tokenizers()
print(f"en vocab size: {EN_TOKENIZER.get_vocab_size().numpy()}")
print(f"pt vocab size: {PT_TOKENIZER.get_vocab_size().numpy()}")

MAX_TOKENS = 128


def _prepare_batch(pt, en):
    pt = PT_TOKENIZER.tokenize(pt)  # Output is ragged.
    pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = EN_TOKENIZER.tokenize(en)
    en = en[:, :(MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return {"pt": pt, "en": en_inputs}, en_labels


TRANSLATE_PREPARE_BATCH_REGISTRY_NAME = "/callable/pt_en_translate/prepare_batch"
gv.GLOBAL_VARIABLE_REPOSITORY.register_callable(
    TRANSLATE_PREPARE_BATCH_REGISTRY_NAME, _prepare_batch)
