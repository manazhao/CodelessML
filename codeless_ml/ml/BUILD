load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

package(
    default_visibility = ["//:__subpackages__"],
)

proto_library(
    name = "configurable_model_proto",
    srcs = ["configurable_model.proto"],
    deps = [
        "//codeless_ml/common:callable_proto",
        "@com_google_protobuf//:any_proto",
    ],
)

py_proto_library(
    name = "configurable_model_py_pb2",
    deps = [":configurable_model_proto"],
)

proto_library(
    name = "input_proto",
    srcs = ["input.proto"],
    deps = ["//codeless_ml/common:callable_proto"],
)

py_proto_library(
    name = "input_py_pb2",
    deps = [
        ":input_proto",
        "//codeless_ml/common:callable_proto",
    ],
)

proto_library(
    name = "train_proto",
    srcs = ["train.proto"],
    deps = [
        ":configurable_model_proto",
        ":input_proto",
    ],
)

py_proto_library(
    name = "train_py_pb2",
    deps = [
        ":configurable_model_proto",
        ":input_proto",
        ":train_proto",
    ],
)

py_library(
    name = "configurable_model",
    srcs = ["configurable_model.py"],
    deps = [
        ":configurable_model_py_pb2",
        "//codeless_ml/common:global_variable",
        "//codeless_ml/ml/transformer:decoder",
        "//codeless_ml/ml/transformer:encoder",
        "//codeless_ml/ml/transformer:positional_embedding",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
        requirement("tensorflow"),
        requirement("tensorflow-hub"),
    ],
)

py_library(
    name = "input",
    srcs = ["input.py"],
    deps = [
        requirement("tensorflow"),
        requirement("tensorflow_datasets"),
        ":input_py_pb2",
        "//codeless_ml/common:global_variable",
        "@abseil-py//absl/logging",
    ],
)

py_library(
    name = "model_trainer",
    srcs = ["model_trainer.py"],
    deps = [
        ":configurable_model",
        ":input",
        ":input_py_pb2",
        ":train_py_pb2",
        "//codeless_ml/common:global_variable",
        "//codeless_ml/ml/registry:modules",
        "//codeless_ml/ml/transformer:loss_and_metric",
        "//codeless_ml/ml/transformer:schedule",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
        requirement("tensorflow"),
        requirement("tensorflow-text"),
    ],
)

py_library(
    name = "setup_mnist",
    testonly = True,
    srcs = ["setup_mnist.py"],
    deps = [
        "//codeless_ml/common:global_variable",
        "@abseil-py//absl/logging",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "configurable_model_test",
    srcs = [
        "configurable_model_test.py",
    ],
    deps = [
        ":configurable_model",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "embedding_model_test",
    srcs = [
        "embedding_model_test.py",
    ],
    deps = [
        ":configurable_model",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
        requirement("tensorflow"),
    ],
)

py_test(
    name = "input_test",
    size = "small",
    srcs = ["input_test.py"],
    deps = [
        ":input",
        requirement("tensorflow"),
        requirement("numpy"),
        "//codeless_ml/common:global_variable",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_test(
    name = "model_trainer_test",
    size = "small",
    srcs = ["model_trainer_test.py"],
    data = ["//codeless_ml/ml/testdata:mnist_testdata"],
    deps = [
        ":model_trainer",
        ":setup_mnist",
        ":train_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        requirement("numpy"),
        requirement("parameterized"),
        requirement("tensorflow"),
        "//codeless_ml/common:global_variable",
    ],
)

py_test(
    name = "transformer_model_test",
    srcs = ["transformer_model_test.py"],
    deps = [
        ":model_trainer",
        ":train_py_pb2",
        "//codeless_ml/common:global_variable",
        "//codeless_ml/ml:input",
        "//codeless_ml/ml/transformer:loss_and_metric",
        "//codeless_ml/ml/transformer:positional_embedding",
        "//codeless_ml/ml/transformer:schedule",
        "@com_google_protobuf//:protobuf_python",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "download_tfds",
    srcs = ["download_tfds.py"],
    deps = [
        requirement("tensorflow_datasets"),
        requirement("tensorflow"),
        "@abseil-py//absl:app",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
    ],
)

py_binary(
    name = "print_tf_record_main",
    srcs = ["print_tf_record_main.py"],
    deps = [
        requirement("tensorflow"),
        "@abseil-py//absl/logging",
    ],
)

py_binary(
    name = "sample_mnist",
    srcs = ["sample_mnist.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
        "@abseil-py//absl/logging",
        requirement("Keras"),
    ],
)

py_binary(
    name = "saved_model_converter_main",
    srcs = ["saved_model_converter_main.py"],
    deps = [
        requirement("tensorflow"),
        "@abseil-py//absl/logging",
    ],
)

py_binary(
    name = "sysinfo",
    srcs = ["sysinfo.py"],
    deps = [requirement("tensorflow")],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    data = [
        "//codeless_ml/ml/artifacts:ted_hrlr_translate",
        "//codeless_ml/ml/artifacts:ted_hrlr_translate_pt_en_converter",
        "//codeless_ml/ml/examples:machine_translation",
    ],
    deps = [
        ":model_trainer",
        ":train_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)
