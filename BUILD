# load("@bazel_gazelle//:def.bzl", "gazelle")
# load("@rules_python_gazelle_plugin//manifest:defs.bzl", "gazelle_python_manifest")
# # Gazelle python extension needs a manifest file mapping from
# # an import to the installed package that provides it.
# # This macro produces two targets:
# # - //:gazelle_python_manifest.update can be used with `bazel run`
# #   to recalculate the manifest
# # - //:gazelle_python_manifest.test is a test target ensuring that
# #   the manifest doesn't need to be updated
# # This target updates a file called gazelle_python.yaml, and
# # requires that file exist before the target is run.
# # When you are using gazelle you need to run this target first.
# gazelle_python_manifest(
#     name = "gazelle_python_manifest",
#     modules_mapping = ":modules_map",
#     pip_repository_name = "pip",
#     requirements = [
#         "//:requirements.txt",
#     ],
#     tags = ["exclusive"],
# )
# 
# # Our gazelle target points to the python gazelle binary.
# # This is the simple case where we only need one language supported.
# # If you also had proto, go, or other gazelle-supported languages,
# # you would also need a gazelle_binary rule.
# # See https://github.com/bazelbuild/bazel-gazelle/blob/master/extend.rst#example
# # This is the primary gazelle target to run, so that you can update BUILD.bazel files.
# # You can execute:
# # - bazel run //:gazelle update
# # - bazel run //:gazelle fix
# # See: https://github.com/bazelbuild/bazel-gazelle#fix-and-update
# gazelle(
#     name = "gazelle",
#     gazelle = "@rules_python_gazelle_plugin//python:gazelle_binary",
# )