load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "modules",
    deps = [
        ":mnist",
        ":mtnt",
    ],
)

py_library(
    name = "mtnt",
    srcs = ["mtnt.py"],
    deps = [
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        "@abseil-py//absl/logging",
        requirement("tensorflow"),
        "//codeless_ml/common:global_variable",
    ],
)

py_test(
    name = "mtnt_test",
    srcs = ["mtnt_test.py"],
    data = [
        "//codeless_ml/ml/artifacts:ted_hrlr_translate",
        "//codeless_ml/ml/artifacts:ted_hrlr_translate_pt_en_converter",
    ],
    deps = [
        ":mtnt",
        "//codeless_ml/ml:input",
        "//codeless_ml/ml:input_py_pb2",
        "@abseil-py//absl/logging",
        "@com_google_protobuf//:protobuf_python",
        requirement("tensorflow_datasets"),
        requirement("tensorflow-text"),
    ],
)

py_test(
    name = "mnist_test",
    srcs = ["mnist_test.py"],
    deps = [
        ":mnist",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)
