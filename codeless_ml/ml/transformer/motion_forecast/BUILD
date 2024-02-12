load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip//:requirements.bzl", "requirement")

package(
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "agent_features",
    srcs = ["agent_features.py"],
    deps = [
        requirement("tensorflow"),
    ],
)

py_library(
    name = "center_agent_features",
    srcs = ["center_agent_features.py"],
    deps = [
        requirement("tensorflow"),
        ":agent_features",
        ":center_agent_util",
    ],
)

py_library(
    name = "center_agent_util",
    srcs = ["center_agent_util.py"],
    deps = [
        requirement("tensorflow"),
    ],
)

py_test(
    name = "agent_features_test",
    srcs = ["agent_features_test.py"],
    deps = [
        ":agent_features",
        ":center_agent_features",
        ":center_agent_util",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "center_agent_util_test",
    srcs = ["center_agent_util_test.py"],
    deps = [
        ":center_agent_util",
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)
