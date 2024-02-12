load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load(
    "@pip//:requirements.bzl",
    "requirement",
)

package(
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    deps = [
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_library(
    name = "decoder",
    srcs = ["decoder.py"],
    deps = [
        ":attention",
        ":ff",
        ":positional_embedding",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "encoder",
    srcs = ["encoder.py"],
    deps = [
        ":attention",
        ":ff",
        ":positional_embedding",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "ff",
    srcs = ["ff.py"],
    deps = [
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_library(
    name = "positional_embedding",
    srcs = ["positional_embedding.py"],
    deps = [
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_library(
    name = "schedule",
    srcs = ["schedule.py"],
    deps = [
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
    ],
)

py_library(
    name = "transformer",
    srcs = ["transformer.py"],
    deps = [
        ":decoder",
        ":encoder",
        requirement("tensorflow"),
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
    name = "attention_test",
    size = "medium",
    srcs = ["attention_test.py"],
    deps = [
        ":attention",
        ":positional_embedding",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "decoder_test",
    size = "medium",
    srcs = ["decoder_test.py"],
    deps = [
        ":decoder",
        ":positional_embedding",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "encoder_test",
    size = "medium",
    srcs = ["encoder_test.py"],
    deps = [
        ":encoder",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "ff_test",
    size = "medium",
    srcs = ["ff_test.py"],
    deps = [
        ":ff",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "loss_and_metric_test",
    srcs = ["loss_and_metric_test.py"],
    deps = [
        ":loss_and_metric",
        "//codeless_ml/common:callable_py_pb2",
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "positional_embedding_test",
    size = "medium",
    srcs = ["positional_embedding_test.py"],
    deps = [
        ":positional_embedding",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "schedule_test",
    size = "medium",
    srcs = ["schedule_test.py"],
    deps = [
        ":schedule",
        "//codeless_ml/common:callable_py_pb2",
        "//codeless_ml/common:global_variable",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)

py_test(
    name = "transformer_test",
    size = "medium",
    srcs = ["transformer_test.py"],
    deps = [
        ":transformer",
        requirement("tensorflow"),
        requirement("numpy"),
    ],
)
