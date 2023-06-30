# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# author: Tomasz Grel (tgrel@nvidia.com)

import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import textwrap
from typing import List

import numpy as np
import tensorflow as tf
from nn.dense_model import DenseModel

from . import constants as c

LOGGER = logging.getLogger(__name__)

_dense_model_config_template = r"""name: "{model_name}"
{backend_type}: "{backend_runtime}"
max_batch_size: 0
input [
  {{
    name: "{input1}"
    data_type: TYPE_FP32
    dims: [-1]
  }},
  {{
    name: "{input2}"
    data_type: TYPE_FP32
    dims: [-1]
  }} 
]
output [
  {{
    name: "{output1}"
    data_type: TYPE_FP32
    dims: [-1,1]
  }}
]
version_policy: {{
        specific:{{versions: 1}}
}},
instance_group [
  {{
    count: {engine_count_per_device}
    kind : KIND_GPU
    gpus: [0]
  }}
]  
"""


def _execute_cmd(cmd: List, verbose: bool = False):
    """Execute command as subprocess.

    Args:
        cmd: A command definition
        verbose: Stream command output

    Raises:
        OSError when command execution failed
    """
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8"
    )

    if verbose:
        LOGGER.info("Command output:")

    stream_output = ""
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            stream_output += output
            if verbose:
                print(textwrap.indent(output.rstrip(), "    "))  # noqa: T201

    result = process.poll()

    if result != 0:
        raise OSError(
            f"Processes exited with error code:{result}. Command to reproduce error:\n{' '.join(cmd)}"
        )


def _savedmodel2onnx(source_model_path, dst_model_path, opset=11, verbose=False):
    convert_cmd = [
        "python",
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        source_model_path.as_posix(),
        "--output",
        dst_model_path.as_posix(),
        "--opset",
        str(opset),
        "--verbose",
    ]

    _execute_cmd(convert_cmd, verbose=verbose)


def _onnx2trt(
    model,
    source_model_path,
    dst_model_path,
    precision,
    optimal_batch_size,
    max_batch_size,
    verbose=False,
):

    min_batch = np.array([model.num_numerical_features, sum(model.embedding_dim)])

    optimal_batch = min_batch * optimal_batch_size
    max_batch = min_batch * max_batch_size

    print(
        f"min batch {min_batch}, optimal_batch: {optimal_batch}, max_batch: {max_batch}"
    )

    convert_cmd = [
        "trtexec",
        f"--onnx={source_model_path.as_posix()}",
        "--buildOnly",
        f"--saveEngine={dst_model_path.as_posix()}",
        f"--minShapes=args_0:{min_batch[0]},args_1:{min_batch[1]}",
        f"--optShapes=args_0:{optimal_batch[0]},args_1:{optimal_batch[1]}",
        f"--maxShapes=args_0:{max_batch[0]},args_1:{max_batch[1]}",
    ]

    if precision == "fp16":
        convert_cmd += ["--fp16"]

    _execute_cmd(convert_cmd, verbose=verbose)


def _convert2onnx(source_model_path, workdir, verbose=False):
    model_path = workdir / "model.onnx"
    _savedmodel2onnx(
        source_model_path=source_model_path,
        dst_model_path=model_path,
        verbose=verbose,
    )
    return model_path


def _convert2trt(
    model,
    source_model_path,
    precision,
    workdir,
    optimal_batch_size,
    max_batch_size,
    verbose=False,
):

    onnx_model_path = _convert2onnx(
        source_model_path=source_model_path,
        workdir=workdir,
        verbose=verbose,
    )
    trt_model_path = workdir / "model.plan"
    _onnx2trt(
        model=model,
        source_model_path=onnx_model_path,
        dst_model_path=trt_model_path,
        precision=precision,
        verbose=verbose,
        optimal_batch_size=optimal_batch_size,
        max_batch_size=max_batch_size,
    )
    return trt_model_path


def _set_tf_memory_growth():
    physical_devices = tf.config.list_physical_devices("GPU")
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)


def deploy_dense(
    src,
    dst,
    model_name,
    model_format,
    model_precision,
    max_batch_size,
    engine_count_per_device,
    trt_optimal_batch_size,
    version="1",
):
    print("deploy dense dst: ", dst)

    _set_tf_memory_growth()

    os.makedirs(dst, exist_ok=True)

    dense_model = DenseModel.from_config(os.path.join(src, "config.json"))
    if model_precision == "fp16" and model_format == 'tf-savedmodel':
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    # Currently, there's no support for custom kernels deployment.
    # Use pure tensorflow implementation instead on the inference side.
    if dense_model.interaction == 'dot_custom_cuda':
        dense_model.interaction = 'dot_tensorflow'
        dense_model._create_interaction_op()

    dense_model.load_weights(os.path.join(src, "dense"))

    # transpose needed here because HPS expects a table-major format vs TensorFlow uses batch-major
    dense_model.transpose = True
    dense_model.force_initialization(training=False, flattened_input=True)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        model_path = tempdir / "model.savedmodel"
        dense_model.save_model(model_path.as_posix(), save_input_signature=False)
        model_store = pathlib.Path(dst) / str(version)
        model_store.mkdir(parents=True, exist_ok=True)

        if model_format == "tf-savedmodel":
            backend_type = "platform"
            backend_runtime = "tensorflow_savedmodel"
            shutil.copytree(model_path, model_store / "model.savedmodel")
        elif model_format == "onnx":
            backend_type = "backend"
            backend_runtime = "onnxruntime"
            model_path = _convert2onnx(model_path, workdir=tempdir)
            shutil.copy(model_path, model_store / "model.onnx")
        elif model_format == "trt":
            backend_type = "backend"
            backend_runtime = "tensorrt"
            model_path = _convert2trt(
                dense_model,
                model_path,
                precision=model_precision,
                workdir=tempdir,
                optimal_batch_size=trt_optimal_batch_size,
                max_batch_size=max_batch_size,
            )
            shutil.copy(model_path, model_store / "model.plan")
        else:
            raise ValueError(f"Unsupported format: {model_format}")

    with open(os.path.join(dst, "config.pbtxt"), "w") as f:
        s = _dense_model_config_template.format(
            backend_type=backend_type,
            backend_runtime=backend_runtime,
            model_name=model_name,
            input1=c.dense_input1_name,
            input2=c.dense_numerical_features_name,
            output1=c.dense_output_name,
            max_batch_size=max_batch_size,
            engine_count_per_device=engine_count_per_device,
        )
        f.write(s)

        print(f"{model_name} configuration:")
        print(s)

