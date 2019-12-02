# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import glob
import inspect
import os
import shutil
import subprocess
from argparse import Namespace
from typing import List, Callable

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _compress(src_path: str, dst_path: str):
    """
    Compress source path into destination path

    :param src_path: (str) Source path
    :param dst_path: (str) Destination path
    """
    print('[*] Compressing...')
    shutil.make_archive(dst_path, 'zip', src_path)
    print('[*] Compressed the contents in: {}.zip'.format(dst_path))


def _print_input(func: Callable):
    """
    Decorator printing function name and args
    :param func: (Callable) Decorated function
    :return: Wrapped call
    """

    def wrapper(*args, **kwargs):
        """
        Print the name and arguments of a function

        :param args: Named arguments
        :param kwargs: Keyword arguments
        :return: Original function call
        """
        tf.logging.set_verbosity(tf.logging.ERROR)
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ''.join('\t{} = {!r}\n'.format(*item) for item in func_args.items())

        print('[*] Running \'{}\' with arguments:'.format(func.__qualname__))
        print(func_args_str[:-1])

        return func(*args, **kwargs)

    return wrapper


def _parse_placeholder_types(values: str):
    """
    Extracts placeholder types from a comma separate list.

    :param values: (str) Placeholder types
    :return: (List) Placeholder types
    """
    values = [int(value) for value in values.split(",")]
    return values if len(values) > 1 else values[0]


def _optimize_checkpoint_for_inference(graph_path: str,
                                       input_names: List[str],
                                       output_names: List[str]):
    """
    Removes Horovod and training related information from the graph

    :param graph_path: (str) Path to the graph.pbtxt file
    :param input_names: (str) Input node names
    :param output_names: (str) Output node names
    """

    print('[*] Optimizing graph for inference ...')

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(graph_path, "rb") as f:
        data = f.read()
        text_format.Merge(data.decode("utf-8"), input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names,
        output_names,
        _parse_placeholder_types(str(dtypes.float32.as_datatype_enum)),
        False)

    print('[*] Saving original graph in: {}'.format(graph_path + '.old'))
    shutil.move(graph_path, graph_path + '.old')

    print('[*] Writing down optimized graph ...')
    graph_io.write_graph(output_graph_def,
                         os.path.dirname(graph_path),
                         os.path.basename(graph_path))


@_print_input
def to_savedmodel(input_shape: str,
                  model_fn: Callable,
                  checkpoint_dir: str,
                  output_dir: str,
                  input_names: List[str],
                  output_names: List[str],
                  use_amp: bool,
                  use_xla: bool,
                  compress: bool,
                  params: Namespace):
    """
    Export checkpoint to Tensorflow savedModel

    :param input_shape: (str) Input shape to the model in format [batch, height, width, channels]
    :param model_fn: (Callable) Estimator's model_fn
    :param checkpoint_dir: (str) Directory where checkpoints are stored
    :param output_dir: (str) Output directory for storage of the generated savedModel
    :param input_names: (List[str]) Input node names
    :param output_names: (List[str]) Output node names
    :param use_amp: (bool )Enable TF-AMP
    :param use_xla: (bool) Enable XLA
    :param compress: (bool) Compress output
    :param params: (Namespace) Namespace to be passed to model_fn
    """
    assert os.path.exists(checkpoint_dir), 'Path not found: {}'.format(checkpoint_dir)
    assert input_shape is not None, 'Input shape must be provided'

    _optimize_checkpoint_for_inference(os.path.join(checkpoint_dir, 'graph.pbtxt'), input_names, output_names)

    try:
        ckpt_path = os.path.splitext([p for p in glob.iglob(os.path.join(checkpoint_dir, '*.index'))][0])[0]
    except IndexError:
        raise ValueError('Could not find checkpoint in directory: {}'.format(checkpoint_dir))

    config_proto = tf.compat.v1.ConfigProto()

    config_proto.allow_soft_placement = True
    config_proto.log_device_placement = False
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.force_gpu_compatible = True

    if use_amp:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
    if use_xla:
        config_proto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    run_config = tf.estimator.RunConfig(
        model_dir=None,
        tf_random_seed=None,
        save_summary_steps=1e9,  # disabled
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        session_config=config_proto,
        keep_checkpoint_max=None,
        keep_checkpoint_every_n_hours=1e9,  # disabled
        log_step_count_steps=1e9,
        train_distribute=None,
        device_fn=None,
        protocol=None,
        eval_distribute=None,
        experimental_distribute=None
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=ckpt_path,
        config=run_config,
        params=params
    )

    print('[*] Exporting the model ...')

    input_type = tf.float16 if use_amp else tf.float32

    def get_serving_input_receiver_fn():

        def serving_input_receiver_fn():
            features = tf.placeholder(dtype=input_type, shape=input_shape, name='input_tensor')

            return tf.estimator.export.TensorServingInputReceiver(features=features, receiver_tensors=features)

        return serving_input_receiver_fn

    export_path = estimator.export_saved_model(
        export_dir_base=output_dir,
        serving_input_receiver_fn=get_serving_input_receiver_fn(),
        checkpoint_path=ckpt_path
    )

    print('[*] Done! path: `%s`' % export_path.decode())

    if compress:
        _compress(export_path.decode(), os.path.join(output_dir, 'saved_model'))


@_print_input
def to_tf_trt(savedmodel_dir: str,
              output_dir: str,
              precision: str,
              feed_dict_fn: Callable,
              num_runs: int,
              output_tensor_names: List[str],
              compress: bool):
    """
    Export Tensorflow savedModel to TF-TRT

    :param savedmodel_dir: (str) Input directory containing a Tensorflow savedModel
    :param output_dir: (str) Output directory for storage of the generated TF-TRT exported model
    :param precision: (str) Desired precision of the network (FP32, FP16 or INT8)
    :param feed_dict_fn: (Callable) Input tensors for INT8 calibration. Model specific.
    :param num_runs: (int) Number of calibration runs.
    :param output_tensor_names: (List) Name of the output tensor for graph conversion. Model specific.
    :param compress: (bool) Compress output
    """
    if savedmodel_dir is None or not os.path.exists(savedmodel_dir):
        raise FileNotFoundError('savedmodel_dir not found: {}'.format(savedmodel_dir))

    if os.path.exists(output_dir):
        print('[*] Output dir \'{}\' is not empty. Cleaning up ...'.format(output_dir))
        shutil.rmtree(output_dir)

    print('[*] Converting model...')

    converter = trt.TrtGraphConverter(input_saved_model_dir=savedmodel_dir,
                                      precision_mode=precision)
    converter.convert()

    if precision == 'INT8':
        print('[*] Running INT8 calibration ...')

        converter.calibrate(fetch_names=output_tensor_names, num_runs=num_runs, feed_dict_fn=feed_dict_fn)

    converter.save(output_dir)

    print('[*] Done! TF-TRT saved_model stored in: `%s`' % output_dir)

    if compress:
        _compress('tftrt_saved_model', output_dir)


@_print_input
def to_onnx(input_dir: str, output_dir: str, compress: bool):
    """
    Convert Tensorflow savedModel to ONNX with tf2onnx

    :param input_dir: (str) Input directory with a Tensorflow savedModel
    :param output_dir: (str) Output directory where to store the ONNX version of the model
    :param compress: (bool) Compress output
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.join(output_dir, 'model.onnx')
    print('[*] Converting model...')

    ret = subprocess.call(['python', '-m', 'tf2onnx.convert',
                           '--saved-model', input_dir,
                           '--output', file_name],
                          stdout=open(os.devnull, 'w'),
                          stderr=subprocess.STDOUT)
    if ret > 0:
        raise RuntimeError('tf2onnx.convert has failed with error: {}'.format(ret))

    print('[*] Done! ONNX file stored in: %s' % file_name)

    if compress:
        _compress(output_dir, 'onnx_model')
