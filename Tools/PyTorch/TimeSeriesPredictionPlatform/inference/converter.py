# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import shutil
import yaml
import conf.conf_utils

from triton.xgboost_triton import run_XGBoost_triton
from omegaconf import OmegaConf
def run_converter(config, export, convert):
    cfg = config
    with open(os.path.join(cfg.checkpoint, ".hydra/config.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    config.inference = cfg
    with open(os.path.join(cfg.checkpoint, ".hydra/config_merged.yaml"), "wb") as f:
        OmegaConf.resolve(config)
        OmegaConf.save(config=config, f=f.name)
    if config.dataset.config.get('xgb', False):
        return run_XGBoost_triton(cfg, config)
    if config.dataset.config.get('stat', False):
        raise ValueError("Stat models not supported in deployment")
    model_name = config.model._target_.split(".")[1]
    precision = cfg.precision
    assert precision in ["fp16", "fp32"], "Precision needs to be either fp32 or fp16"
    # export model
    output_path = os.path.join(cfg.checkpoint, "deployment")
    os.makedirs(output_path, exist_ok=True)
    tspp_main_dir = os.path.sep + os.path.join(*(os.getcwd().split(os.path.sep)[:-3]))
    model_format = "torchscript" if export.config.type != "onnx" else export.config.type

    subprocess.run(
        [
            "python",
            "triton/export_model.py",
            "--input-path",
            "triton/model.py",
            "--input-type",
            "pyt",
            "--output-path",
            "{}/exported_model.pt".format(output_path),
            "--output-type",
            "{}".format(export.config.type),
            "--dataloader",
            "triton/dataloader.py",
            "--batch-size",
            "{}".format(cfg.batch_size),
            "--model-dir",
            "{}".format(cfg.checkpoint),
            "--onnx-opset",
            "13",
            "--ignore-unknown-parameters",
        ],
        cwd=tspp_main_dir,
        check=True,
    )
    if model_format == "torchscript":
        with open(output_path + "/exported_model.pt.yaml", "r") as stream:
            var_config = yaml.safe_load(stream)
        var_config_list = []
        for arg in ["--value-ranges", "--max-shapes", "--dtypes", "--min-shapes"]:
            var_config_list.append(arg)
            if arg == "--value-ranges":
                for k, v in var_config["inputs"].items():
                    var_config_list.append(k + "=0,1")
            elif arg == "--max-shapes":
                for k, v in var_config["inputs"].items():
                    var_config_list.append(k + "=" + ",".join([str(cfg.batch_size)] + [str(x) for x in v["shape"][1:]]))
            elif arg == "--min-shapes":
                for k, v in var_config["inputs"].items():
                    var_config_list.append(k + "=" + ",".join([str(x) for x in v["shape"]]))
            else:
                for k, v in var_config["inputs"].items():
                    var_config_list.append(k + "=" + v["dtype"])
    else:
        var_config_list = []
    # model-navigator run
    if cfg.optimize:
        subprocess.run(
            [
                "model-navigator",
                "run",
                "--model-name",
                model_name,
                "--model-path",
                "{}/exported_model.pt".format(output_path),
                "--config-path",
                "{}/exported_model.pt.yaml".format(output_path),
                "--override-workspace",
                "--workspace-path",
                "{}/navigator_workspace".format(output_path),
                "--verbose",
                "--target-formats",
                "{}".format(convert.config.type),
                "--model-format",
                model_format,
                "--config-search-concurrency",
                "1",
                "32",
                "1024",
                "--triton-launch-mode",
                "docker",
                "--max-workspace-size",
                "10000000000",
                "--max-batch-size",
                "{}".format(cfg.batch_size),
                "--gpus",
                "{}".format(cfg.gpu),
                "--atol",
                "1e-3",
                "--rtol",
                "100",
                "--onnx-opsets",
                "13",
                "--container-version",
                "21.12",
            ]
            + var_config_list,
            cwd=tspp_main_dir,
            check=True,
        )
    else:
        subprocess.run(
            [
                "model-navigator",
                "convert",
                "--model-name",
                model_name,
                "--model-path",
                "{}/exported_model.pt".format(output_path),
                "--override-workspace",
                "--workspace-path",
                "{}/navigator_workspace".format(output_path),
                "--output-path",
                "{}/converted_model".format(output_path),
                "--verbose",
                "--target-formats",
                "{}".format(convert.config.type),
                "--model-format",
                model_format,
                "--launch-mode",
                "local",
                "--max-workspace-size",
                "10000000000",
                "--max-batch-size",
                "{}".format(cfg.batch_size),
                "--target-precisions",
                precision,
                "--gpus",
                "{}".format(cfg.gpu),
                "--atol",
                "1e-3",
                "--rtol",
                "100",
                "--onnx-opsets",
                "13",
                "--container-version",
                "21.12",
            ]
            + var_config_list,
            cwd=tspp_main_dir,
            check=True,
        )
        subprocess.run(
            [
                "model-navigator",
                "triton-config-model",
                "--model-name",
                model_name,
                "--model-path",
                "{}/converted_model".format(output_path),
                "--model-version",
                "1",
                "--model-format",
                "{}".format(convert.config.type),
                "--model-repository",
                "{}/navigator_workspace/model-store/".format(output_path),
                "--backend-accelerator",
                cfg.accelerator,
                "--max-batch-size",
                "{}".format(cfg.batch_size),
                "--engine-count-per-device",
                "gpu=2",
                "--tensorrt-precision",
                precision,
                "--tensorrt-capture-cuda-graph",
                "--verbose",
            ],
            cwd=tspp_main_dir,
            check=True,
        )
        convert_type = (
            convert.config.type if convert.config.type != "torchscript" else export.config.type
        )
        subprocess.run(
            [
                "python",
                "triton/check_accuracy.py",
                "--native-model",
                cfg.checkpoint,
                "--native-type",
                "pyt",
                "--export-model",
                "{}/exported_model.pt".format(output_path),
                "--export-type",
                export.config.type,
                "--convert-model",
                "{}/converted_model".format(output_path),
                "--convert-type",
                convert_type,
                "--dataloader",
                "triton/dataloader.py",
                "--batch-size",
                "{}".format(1),
                "--model-dir",
                "{}".format(cfg.checkpoint),
            ],
            cwd=tspp_main_dir,
            check=True,
        )
