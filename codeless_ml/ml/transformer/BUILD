load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load(
    "@pip//:requirements.bzl",
    "requirement",
)

package(
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "model",
    srcs = [":model.py"],
    deps = [
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_library(
    name = "loss_and_metric",
    srcs = ["loss_and_metric.py"],
    deps = [
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "model_test",
    size = "small",
    srcs = ["model_test.py"],
    deps = [
        ":model",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "loss_and_metric_test",
    srcs = ["loss_and_metric_test.py"],
    deps = [
        ":loss_and_metric",
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)
