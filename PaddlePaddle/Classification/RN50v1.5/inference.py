# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import os
import time
import glob

import numpy as np
import dllogger

from paddle.fluid import LoDTensor
from paddle.inference import Config, PrecisionType, create_predictor

from dali import dali_dataloader, dali_synthetic_dataloader
from utils.config import parse_args, print_args
from utils.mode import Mode
from utils.logger import setup_dllogger


def init_predictor(args):
    infer_dir = args.inference_dir
    assert os.path.isdir(
        infer_dir), f'inference_dir = "{infer_dir}" is not a directory'
    pdiparams_path = glob.glob(os.path.join(infer_dir, '*.pdiparams'))
    pdmodel_path = glob.glob(os.path.join(infer_dir, '*.pdmodel'))
    assert len(pdiparams_path) == 1, \
        f'There should be only 1 pdiparams in {infer_dir}, but there are {len(pdiparams_path)}'
    assert len(pdmodel_path) == 1, \
        f'There should be only 1 pdmodel in {infer_dir}, but there are {len(pdmodel_path)}'
    predictor_config = Config(pdmodel_path[0], pdiparams_path[0])
    predictor_config.enable_memory_optim()
    predictor_config.enable_use_gpu(0, args.device)
    precision = args.precision
    max_batch_size = args.batch_size
    assert precision in ['FP32', 'FP16', 'INT8'], \
        'precision should be FP32/FP16/INT8'
    if precision == 'INT8':
        precision_mode = PrecisionType.Int8
    elif precision == 'FP16':
        precision_mode = PrecisionType.Half
    elif precision == 'FP32':
        precision_mode = PrecisionType.Float32
    else:
        raise NotImplementedError
    predictor_config.enable_tensorrt_engine(
        workspace_size=args.workspace_size,
        max_batch_size=max_batch_size,
        min_subgraph_size=args.min_subgraph_size,
        precision_mode=precision_mode,
        use_static=args.use_static,
        use_calib_mode=args.use_calib_mode)
    predictor_config.set_trt_dynamic_shape_info(
        {"data": (1,) + tuple(args.image_shape)},
        {"data": (args.batch_size,) + tuple(args.image_shape)},
        {"data": (args.batch_size,) + tuple(args.image_shape)},
    )
    predictor = create_predictor(predictor_config)
    return predictor


def predict(predictor, input_data):
    '''
    Args:
        predictor: Paddle inference predictor
        input_data: A list of input
    Returns:
        output_data: A list of output
    '''
    # copy image data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)

        if isinstance(input_data[i], LoDTensor):
            input_tensor.share_external_data(input_data[i])
        else:
            input_tensor.reshape(input_data[i].shape)
            input_tensor.copy_from_cpu(input_data[i])

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results


def benchmark_dataset(args):
    """
    Benchmark DALI format dataset, which reflects real the pipeline throughput including
    1. Read images
    2. Pre-processing
    3. Inference
    4. H2D, D2H
    """
    predictor = init_predictor(args)

    dali_iter = dali_dataloader(args, Mode.EVAL, 'gpu:' + str(args.device))

    # Warmup some samples for the stable performance number
    batch_size = args.batch_size
    image_shape = args.image_shape
    images = np.zeros((batch_size, *image_shape)).astype(np.float32)
    for _ in range(args.benchmark_warmup_steps):
        predict(predictor, [images])[0]

    total_images = 0
    correct_predict = 0

    latency = []

    start = time.perf_counter()
    last_time_step = time.perf_counter()
    for dali_data in dali_iter:
        for data in dali_data:
            label = np.asarray(data['label'])
            total_images += label.shape[0]
            label = label.flatten()
            images = data['data']
            predict_label = predict(predictor, [images])[0]
            correct_predict += (label == predict_label).sum()
        batch_end_time_step = time.perf_counter()
        batch_latency = batch_end_time_step - last_time_step
        latency.append(batch_latency)
        last_time_step = time.perf_counter()
    end = time.perf_counter()

    latency = np.array(latency) * 1000
    quantile = np.quantile(latency, [0.9, 0.95, 0.99])

    statistics = {
        'precision': args.precision,
        'batch_size': batch_size,
        'throughput': total_images / (end - start),
        'accuracy': correct_predict / total_images,
        'eval_latency_avg': np.mean(latency),
        'eval_latency_p90': quantile[0],
        'eval_latency_p95': quantile[1],
        'eval_latency_p99': quantile[2],
    }
    return statistics


def benchmark_synthetic(args):
    """
    Benchmark on the synthetic data and bypass all pre-processing.
    The host to device copy is still included.
    This used to find the upper throughput bound when tunning the full input pipeline.
    """

    predictor = init_predictor(args)
    dali_iter = dali_synthetic_dataloader(args, 'gpu:' + str(args.device))

    batch_size = args.batch_size
    image_shape = args.image_shape
    images = np.random.random((batch_size, *image_shape)).astype(np.float32)

    latency = []

    # warmup
    for _ in range(args.benchmark_warmup_steps):
        predict(predictor, [images])[0]

    # benchmark
    start = time.perf_counter()
    last_time_step = time.perf_counter()
    for dali_data in dali_iter:
        for data in dali_data:
            images = data['data']
            predict(predictor, [images])[0]
        batch_end_time_step = time.perf_counter()
        batch_latency = batch_end_time_step - last_time_step
        latency.append(batch_latency)
        last_time_step = time.perf_counter()
    end = time.perf_counter()

    latency = np.array(latency) * 1000
    quantile = np.quantile(latency, [0.9, 0.95, 0.99])

    statistics = {
        'precision': args.precision,
        'batch_size': batch_size,
        'throughput': args.benchmark_steps * batch_size / (end - start),
        'eval_latency_avg': np.mean(latency),
        'eval_latency_p90': quantile[0],
        'eval_latency_p95': quantile[1],
        'eval_latency_p99': quantile[2],
    }
    return statistics

def main(args):
    setup_dllogger(args.report_file)
    if args.show_config:
        print_args(args)

    if args.use_synthetic:
        statistics = benchmark_synthetic(args)
    else:
        statistics = benchmark_dataset(args)

    dllogger.log(step=tuple(), data=statistics)


if __name__ == '__main__':
    main(parse_args(script='inference'))
