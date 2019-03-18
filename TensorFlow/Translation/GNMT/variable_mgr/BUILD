licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "variable_mgr_util",
    srcs = [
        "variable_mgr_util.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:data_flow_ops",
        "//third_party/tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "allreduce",
    srcs = [
        "allreduce.py",
    ],
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/contrib/all_reduce",
        "//third_party/tensorflow/python:collective_ops",
        "//third_party/tensorflow/python:nccl_ops",
    ],
)

py_library(
    name = "batch_allreduce",
    srcs = [
        "batch_allreduce.py",
    ],
    deps = [
        ":allreduce",
        ":constants",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "variable_mgr",
    srcs = [
        "variable_mgr.py",
    ],
    deps = [
        ":allreduce",
        ":batch_allreduce",
        ":variable_mgr_util",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "constants",
    srcs = [
        "constants.py",
    ],
)
