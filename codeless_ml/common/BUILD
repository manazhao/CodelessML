load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//:__subpackages__"],
)

proto_library(
    name = "callable_proto",
    srcs = ["callable.proto"],
)

py_proto_library(
    name = "callable_py_pb2",
    deps = [":callable_proto"],
)

py_library(
    name = "global_variable",
    srcs = ["global_variable.py"],
    deps = [
        ":callable_py_pb2",
        "@abseil-py//absl/logging",
    ],
)

py_test(
    name = "global_variable_test",
    srcs = ["global_variable_test.py"],
    deps = [":global_variable"],
)