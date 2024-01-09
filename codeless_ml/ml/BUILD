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
    deps = ["//codeless_ml/common:callable_proto", "@com_google_protobuf//:any_proto"],
)

py_proto_library(
    name = "configurable_model_py_pb2",
    deps = [":configurable_model_proto"]
)

proto_library(
    name = "input_proto",
    srcs = ["input.proto"],
    deps = ["//codeless_ml/common:callable_proto"],
)

py_proto_library(
    name = "input_py_pb2",
    deps = ["//codeless_ml/common:callable_proto", ":input_proto"],
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
        ":train_proto"
    ],
)

py_library(
    name = "configurable_model",
    srcs = ["configurable_model.py"],
    deps = [
        "@abseil-py//absl/logging",
        requirement("tensorflow"),
        ":configurable_model_py_pb2",
        "@com_google_protobuf//:protobuf_python",
        "//codeless_ml/common:global_variable",
    ],
)

py_library(
    name = "input",
    srcs = ["input.py"],
    deps = [
        requirement("tensorflow"),
        ":input_py_pb2",
        "//codeless_ml/common:global_variable",
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
        ":trainer_modules",
        "@com_google_protobuf//:protobuf_python",
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
        "@abseil-py//absl/logging",
    ],
)

py_library(
    name = "register_callable",
    srcs = ["register_callable.py"],
    deps = [
        "@abseil-py//absl/logging",
        requirement("tensorflow"),
        "//codeless_ml/common:global_variable",
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

# A list of modules which will be loaded by the trainer before the training. In
# the module, the user can setup the necessary context for the training.
py_library(
    name = "trainer_modules",
    srcs = [
        ":register_callable",
    ],
)

py_test(
    name = "configurable_model_test",
    srcs = [
        "configurable_model_test.py",
    ],
    deps = [
        ":configurable_model",
        "@com_google_protobuf//:protobuf_python",
        "@abseil-py//absl/logging",
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
        "@com_google_protobuf//:protobuf_python",
        "@abseil-py//absl/logging",
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
    name = "register_callable_test",
    srcs = ["register_callable_test.py"],
    deps = [
        ":register_callable",
        requirement("numpy"),
        requirement("tensorflow"),
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
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":model_trainer",
        ":train_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)

