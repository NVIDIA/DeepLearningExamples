# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

DEFAULT_DIR = "/outbrain"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tensorflow2 WideAndDeep Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=True,
    )

    locations = parser.add_argument_group("location of datasets")

    locations.add_argument(
        "--dataset_path",
        type=str,
        default=f"{DEFAULT_DIR}/data",
        help="Dataset base directory, relative to which path to feature_spec and paths in feature_spec are resolved"
    )

    locations.add_argument(
        "--fspec_file",
        type=str,
        default="feature_spec.yaml",
        help="Path to the feature spec file, relative to dataset_path"
    )

    locations.add_argument(
        "--embedding_sizes_file",
        type=str,
        default="data/outbrain/embedding_sizes.json",
        help="Path to the file containing a dictionary of embedding sizes for categorical features"
    )

    locations.add_argument(
        "--use_checkpoint",
        default=False,
        action="store_true",
        help="Use checkpoint stored in model_dir path",
    )

    locations.add_argument(
        "--model_dir",
        type=str,
        default=f"{DEFAULT_DIR}/checkpoints",
        help="Destination where the model checkpoint will be saved",
    )

    locations.add_argument(
        "--results_dir",
        type=str,
        default="/results",
        help="Directory to store training results",
    )

    locations.add_argument(
        "--log_filename",
        type=str,
        default="log.json",
        help="Name of the file to store dlloger output",
    )

    training_params = parser.add_argument_group("training parameters")

    training_params.add_argument(
        "--global_batch_size",
        type=int,
        default=131072,
        help="Total (global) size of training batch",
    )

    training_params.add_argument(
        "--eval_batch_size",
        type=int,
        default=131072,
        help="Total (global) size of evaluation batch",
    )

    training_params.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs"
    )

    training_params.add_argument(
        "--cpu", default=False, action="store_true", help="Run computations on the CPU"
    )

    training_params.add_argument(
        "--amp",
        default=False,
        action="store_true",
        help="Enable automatic mixed precision conversion",
    )

    training_params.add_argument(
        "--xla", default=False, action="store_true", help="Enable XLA conversion"
    )

    training_params.add_argument(
        "--linear_learning_rate",
        type=float,
        default=0.02,
        help="Learning rate for linear model",
    )

    training_params.add_argument(
        "--deep_learning_rate",
        type=float,
        default=0.00012,
        help="Learning rate for deep model",
    )

    training_params.add_argument(
        "--deep_warmup_epochs",
        type=float,
        default=6,
        help="Number of learning rate warmup epochs for deep model",
    )

    model_construction = parser.add_argument_group("model construction")

    model_construction.add_argument(
        "--deep_hidden_units",
        type=int,
        default=[1024, 1024, 1024, 1024, 1024],
        nargs="+",
        help="Hidden units per layer for deep model, separated by spaces",
    )

    model_construction.add_argument(
        "--deep_dropout",
        type=float,
        default=0.1,
        help="Dropout regularization for deep model",
    )

    model_construction.add_argument(
        "--combiner",
        type=str,
        default="sum",
        choices=[
            "mean",
            "sum",
        ],
        help="Type of aggregation used for multi hot categorical features",
    )

    run_params = parser.add_argument_group("run mode parameters")

    run_params.add_argument(
        "--num_auc_thresholds",
        type=int,
        default=8000,
        help="Number of thresholds for the AUC computation",
    )

    run_params.add_argument(
        "--disable_map_calculation",
        dest="map_calculation_enabled",
        action="store_false",
        default=True,
        help="Disable calculation of MAP metric. See ReadMe for additional dataset requirements keeping it enabled introduces."
    )

    run_params.add_argument(
        "--evaluate",
        default=False,
        action="store_true",
        help="Only perform an evaluation on the validation dataset, don't train",
    )

    run_params.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run training or evaluation benchmark to collect performance metrics",
    )

    run_params.add_argument(
        "--benchmark_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps before the start of the benchmark",
    )

    run_params.add_argument(
        "--benchmark_steps",
        type=int,
        default=1000,
        help="Number of steps for performance benchmark",
    )

    run_params.add_argument(
        "--affinity",
        type=str,
        default="unique_interleaved",
        choices=[
            "all",
            "single",
            "single_unique",
            "unique_interleaved",
            "unique_contiguous",
            "disabled",
        ],
        help="Type of CPU affinity",
    )

    return parser.parse_args()
