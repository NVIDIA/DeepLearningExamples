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


import argparse
import os

import tensorflow as tf
import horovod.tensorflow as hvd

import deployment.tf
import deployment.hps


def clear_and_create_directory(repo_path):
    print("creating directory:", repo_path)
    os.makedirs(repo_path, exist_ok=True)


def create_model_repo(dst, sparse_model_name, dense_model_name, ensemble_name):
    clear_and_create_directory(dst)
    created = []
    for name in sparse_model_name, dense_model_name, ensemble_name:
        d = os.path.join(dst, name)
        clear_and_create_directory(d)
        created.append(d)
    return created


def set_tf_memory_growth():
    physical_devices = tf.config.list_physical_devices("GPU")
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--checkpoint-dir", type=str, help="Source directory with a checkpoint"
    )
    parser.add_argument(
        "--model-repository-path",
        type=str,
        help="Destination directory with Triton model repository",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="The name of the model used for inference.",
        required=True,
    )
    parser.add_argument(
        "--sparse-model-name",
        type=str,
        default='sparse'
    )
    parser.add_argument(
        "--dense-model-name",
        type=str,
        default='dense'
    )
    parser.add_argument(
        "--model-version",
        type=int,
        help="The version of the model used for inference.",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--dense-format",
        type=str,
        help="Target format of dense model part in ensemble.",
        choices=["tf-savedmodel", "onnx", "trt"],
        required=True,
        default="tf-savedmodel",
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
        "--model-precision",
        type=str,
        help="Target precision of dense model part in ensemble.",
        choices=["fp16", "fp32"],
        required=True,
        default="fp32",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="The maximal batch size for deployed model.",
        required=False,
        default=32768,
    )
    parser.add_argument(
        "--trt-optimal-batch-size",
        type=int,
        help="Batch size to optimize TensorRT performance for.",
        required=False,
        default=1024,
    )
    parser.add_argument(
        "--memory-threshold-gb",
        type=int,
        help="Amount of memory in GB after reaching which CPU offloading will be used",
        required=False,
        default=70,
    )
    parser.add_argument(
        "--engine-count-per-device",
        type=int,
        default=1,
        help="Number of model instances per GPU",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to deploy HPS onto",
    )
    parser.add_argument(
        "--fused_embedding",
        action="store_true",
        default=False,
        help="Fuse the embedding table together for better GPU utilization.",
    )
    parser.add_argument(
        "--hps_gpucacheper",
        type=float,
        default=0.25,
        help="Fraction of the embeddings to store in GPU cache.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="grpc://127.0.0.1:8001",
        help="Url of Triton Inference Server",
        required=False,
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=False,
        help="Call load model Triton endpoint after creating model store.",
    )
    parser.add_argument(
        "--load-model-timeout-s",
        type=int,
        default=120,
        help="Timeout of load model operation.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Run the entire model on CPU",
    )
    parser.add_argument(
        "--monolithic",
        action="store_true",
        default=False,
        help="Don't use the ensemble paradigm. Instead, save everything into a single large SavedModel file",
    )
    args = parser.parse_args()

    hvd.init()

    set_tf_memory_growth()

    deployment_package = deployment.hps if args.sparse_format == 'hps' else deployment.tf

    if args.monolithic:
        deployment_package.deploy_monolithic(sparse_src=os.path.join(args.checkpoint_dir, "sparse"),
            dense_src=os.path.join(args.checkpoint_dir, "dense"),
            dst=args.model_repository_path,
            model_name='dlrm',
            max_batch_size=65536,
            engine_count_per_device=1,
            num_gpus=1,
            version="1",
            cpu=args.cpu,
            model_precision='fp32')
        return

    sparse_dst, dense_dst, ensemble_dst = create_model_repo(
        dst=args.model_repository_path, ensemble_name=args.model_name,
        sparse_model_name=args.sparse_model_name, dense_model_name=args.dense_model_name
    )

    num_numerical_features = deployment_package.deploy_dense(
        src=os.path.join(args.checkpoint_dir, "dense"),
        dst=dense_dst,
        model_name=args.dense_model_name,
        model_format=args.dense_format,
        model_precision=args.model_precision,
        max_batch_size=args.max_batch_size,
        trt_optimal_batch_size=args.trt_optimal_batch_size,
        engine_count_per_device=args.engine_count_per_device,
    )
    num_cat_features = deployment_package.deploy_sparse(
        src=os.path.join(args.checkpoint_dir, "sparse"),
        dst=sparse_dst,
        model_name=args.sparse_model_name,
        num_gpus=args.num_gpus,
        fused=args.fused_embedding,
        max_batch_size=args.max_batch_size,
        gpucacheper=args.hps_gpucacheper,
        engine_count_per_device=args.engine_count_per_device,
        memory_threshold_gb=args.memory_threshold_gb
    )
    deployment_package.deploy_ensemble(
        dst=ensemble_dst,
        model_name=args.model_name,
        sparse_model_name=args.sparse_model_name,
        dense_model_name=args.dense_model_name,
        num_cat_features=num_cat_features,
        num_numerical_features=num_numerical_features,
        version=args.model_version,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
