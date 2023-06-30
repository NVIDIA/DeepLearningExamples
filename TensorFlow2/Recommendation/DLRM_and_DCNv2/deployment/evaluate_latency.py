#!/usr/bin/env python3

# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import json
import logging
import os
import pathlib
import base64

import tensorflow as tf
import numpy as np

# method from PEP-366 to support relative import in executed modules
if __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

import dataloading.feature_spec
from dataloading.dataloader import create_input_pipelines, get_dataset_metadata

from deployment.hps import constants
from deployment.hps.triton_ensemble_wrapper import NumpyToHpsInputConverter

from deployment.deployment_toolkit.core import EvaluationMode, MeasurementMode, OfflineMode
from deployment.deployment_toolkit.triton_performance_runner import TritonPerformanceRunner

LOGGER = logging.getLogger("run_performance_on_triton")


def b64_tensor(x):
    return {'b64': base64.b64encode(x.flatten()).decode("utf-8")}


def create_input_data(sparse_backend, *args, **kwargs):

    if sparse_backend == 'hps':
        return create_input_data_hps(*args, **kwargs)
    elif sparse_backend == 'tf-savedmodel':
        return create_input_data_tf(*args, **kwargs)
    else:
        raise ValueError(f'Unknown sparse backend: {sparse_backend}')


def create_input_data_tf(batch_sizes, dataset_path, dataset_type, feature_spec,
                         total_benchmark_samples, fused_embedding):
    fspec = dataloading.feature_spec.FeatureSpec.from_yaml(
        os.path.join(dataset_path, feature_spec)
    )
    num_tables = len(fspec.get_categorical_sizes())
    table_ids = list(range(num_tables))

    filename = f"/tmp/triton_input_data_batch.json"
    print("generating input data: ", filename)

    _, dataloader = create_input_pipelines(dataset_type=dataset_type, dataset_path=dataset_path, train_batch_size=1,
                                           test_batch_size=1, table_ids=table_ids, feature_spec=feature_spec,
                                           rank=0, world_size=1)
    generated = 0
    samples = []
    for sample in dataloader.op():
        features, labels = sample
        numerical_features, cat_features = features

        cat_features = tf.concat(cat_features, axis=1).numpy().astype(np.int32)
        numerical_features = numerical_features.numpy().astype(np.float32)

        sample = {
            "categorical_features": b64_tensor(cat_features),
            "numerical_features": b64_tensor(numerical_features),
        }
        samples.append(sample)
        generated += 1
        if generated >= total_benchmark_samples:
            break

    with open(filename, "w") as f:
        json.dump(obj={"data": samples}, fp=f, indent=4)

    shapes = [
        f"categorical_features:{cat_features.shape[1]}",
        f"numerical_features:{numerical_features.shape[1]}",
    ]

    input_data = {}
    for batch_size in batch_sizes:
        input_data[batch_size] = (filename, shapes)
    return input_data


def create_input_data_hps(batch_sizes, dataset_path, dataset_type, feature_spec,
                          total_benchmark_samples, fused_embedding):

    input_data = {}
    for batch_size in batch_sizes:
        filename = f"/tmp/triton_input_data_batch{batch_size}.json"
        print("generating input data: ", filename)
        shapes = create_input_data_hps_batch(batch_size=batch_size, dst_path=filename, dataset_path=dataset_path,
                                             dataset_type=dataset_type, feature_spec=feature_spec,
                                             total_benchmark_samples=total_benchmark_samples,
                                             fused_embedding=fused_embedding)
        input_data[batch_size] = (filename, shapes)
    return input_data


def create_input_data_hps_batch(batch_size, dst_path, dataset_path, dataset_type, feature_spec,
                      total_benchmark_samples, fused_embedding):

    fspec = dataloading.feature_spec.FeatureSpec.from_yaml(
        os.path.join(dataset_path, feature_spec)
    )
    num_tables = len(fspec.get_categorical_sizes())
    table_ids = list(range(num_tables))

    converter = NumpyToHpsInputConverter(categorical_sizes=fspec.get_categorical_sizes(),
                                         fused_embedding=fused_embedding)

    _, dataloader = create_input_pipelines(dataset_type=dataset_type, dataset_path=dataset_path,
                                           train_batch_size=batch_size, test_batch_size=batch_size,
                                           table_ids=table_ids, feature_spec=feature_spec, rank=0, world_size=1)

    generated = 0
    batches = []
    for batch in dataloader.op():
        features, labels = batch
        numerical_features, cat_features = features
        key_tensor, nkey_tensor, numerical_features = converter(
            numerical_features, cat_features
        )

        batch = {
            constants.key_global_prefix: b64_tensor(key_tensor),
            constants.numkey_global_prefix: b64_tensor(nkey_tensor),
            constants.ens_numerical_features_name: b64_tensor(numerical_features)
        }
        batches.append(batch)
        generated += batch_size
        if generated >= total_benchmark_samples:
            break

    with open(dst_path, "w") as f:
        json.dump(obj={"data": batches}, fp=f, indent=4)

    shapes = [
        f"{constants.key_global_prefix}:{key_tensor.shape[1]}",
        f"{constants.numkey_global_prefix}:{nkey_tensor.shape[1]}",
        f"{constants.ens_numerical_features_name}:{numerical_features.shape[1]}",
    ]
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to test",
    )
    parser.add_argument(
        "--result-path",
        type=pathlib.Path,
        required=True,
        help="Path where results files is stored.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Url to Triton server",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default=1,
        help="Version of model",
    )
    parser.add_argument(
        "--sparse-format",
        type=str,
        help="Target format of dense model part in ensemble.",
        choices=["tf-savedmodel", "hps"],
        required=True,
        default="tf-savedmodel",
    )
    parser.add_argument(
        "--fused-embedding",
        action="store_true",
        help="Use the fused embedding API for HPS",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="List of batch sizes to test.",
        nargs="*",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=[1],
        help="List of concurrency modes.",
        nargs="*",
    )
    parser.add_argument(
        "--measurement-mode",
        choices=[item.value for item in MeasurementMode],
        default=MeasurementMode.COUNT_WINDOWS.value,
        type=str,
        help="Select measurement mode "
        "'time_windows' stabilize performance on measurement window. "
        "'count_windows' stabilize performance on number of samples.",
    )
    parser.add_argument(
        "--measurement-interval",
        help="Time window perf_analyzer will wait to stabilize the measurement",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--measurement-request-count",
        help="Number of samples on which perf_analyzer will stabilize the measurement",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=[item.value for item in EvaluationMode],
        default=EvaluationMode.OFFLINE.value,
        type=str,
        help="Select evaluation mode "
        "'offline' run offline analysis and use GPU memory to pass tensors. "
        "'online' run online analysis and use HTTP protocol.",
    )
    parser.add_argument(
        "--offline-mode",
        choices=[item.value for item in OfflineMode],
        default=OfflineMode.SYSTEM.value,
        type=str,
        help="Select offline mode "
        "'system' pass tensors through CPU RAM memory. "
        "'cuda' pass tensors through GPU RAM memory.",
    )
    parser.add_argument(
        "--output-shared-memory-size",
        default=524288,
        type=int,
        help="Size of memory buffer allocated for output with dynamic shapes in bytes. "
        "Has to be equal to maximal size of output tensor.",
    )
    parser.add_argument(
        "--warmup",
        help="Enable model warmup before performance test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--timeout",
        help="Timeout for performance analysis",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose logs",
        action="store_true",
        default=False,
    )

    # dataset and dataloading settings
    parser.add_argument(
        "--dataset_path", default=None, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--feature_spec",
        default="feature_spec.yaml",
        help="Name of the feature spec file in the dataset directory",
    )
    parser.add_argument(
        "--dataset_type",
        default="tf_raw",
        choices=["tf_raw", "synthetic", "split_tfrecords"],
        help="The type of the dataset to use",
    )

    parser.add_argument(
        "--num-benchmark-samples",
        default=2**18,
        type=int,
        help="The type of the dataset to use",
    )

    args = parser.parse_args()

    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    input_data = create_input_data(sparse_backend=args.sparse_format,
                                   batch_sizes=args.batch_sizes, dataset_path=args.dataset_path,
                                   dataset_type=args.dataset_type, feature_spec=args.feature_spec,
                                   total_benchmark_samples=args.num_benchmark_samples,
                                   fused_embedding=args.fused_embedding)

    runner = TritonPerformanceRunner(
        server_url=args.server_url,
        model_name=args.model_name,
        input_data=input_data,
        batch_sizes=args.batch_sizes,
        measurement_mode=MeasurementMode(args.measurement_mode),
        measurement_interval=args.measurement_interval,
        measurement_request_count=args.measurement_request_count,
        concurrency=args.concurrency,
        evaluation_mode=EvaluationMode(args.evaluation_mode),
        offline_mode=OfflineMode(args.offline_mode),
        output_shared_memory_size=args.output_shared_memory_size,
        result_path=args.result_path,
        warmup=args.warmup,
        timeout=args.timeout,
        verbose=args.verbose,
        flattened_input=args.sparse_format == 'hps'
    )

    runner.run()

    for _, (filename, _) in input_data.items():
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == "__main__":
    main()
