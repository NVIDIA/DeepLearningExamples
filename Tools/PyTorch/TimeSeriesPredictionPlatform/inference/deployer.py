# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from typing import Dict, List, Optional, Tuple

import yaml
from omegaconf import OmegaConf

from conf.conf_utils import append_derived_config_fields


def run_deployment(config):
    inference_dir = os.getcwd()
    cfg = config
    with open(os.path.join(cfg.evaluator.checkpoint, ".hydra/config.yaml"), "rb") as f:
        config = OmegaConf.load(f)
        append_derived_config_fields(config)
    config.config.evaluator = OmegaConf.merge(config.config.evaluator, cfg.evaluator)
    if cfg.inference.get("dataset_dir", None):
        config.config.dataset.dest_path = cfg.inference.dataset_dir
    with open(os.path.join(cfg.evaluator.checkpoint, ".hydra/config_merged.yaml"), "wb") as f:
        OmegaConf.save(config=config, f=f.name)
    model_name = config.config.model._target_.split(".")[1]
    precision = cfg.inference.precision
    assert precision in ["fp16", "fp32"], "Precision needs to be either fp32 or fp16"
    # export model
    output_path = os.path.join(cfg.evaluator.checkpoint, "deployment")
    os.makedirs(output_path, exist_ok=True)
    tspp_main_dir = os.path.sep + os.path.join(*(os.getcwd().split(os.path.sep)[:-3]))
    model_format = "torchscript" if cfg.inference.export.type != "onnx" else cfg.inference.export.type

    if not cfg.inference.skip_conversion and not cfg.inference.just_deploy:
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
                "{}".format(cfg.inference.export.type),
                "--dataloader",
                "triton/dataloader.py",
                "--batch-size",
                "{}".format(cfg.inference.batch_size),
                "--model-dir",
                "{}".format(cfg.evaluator.checkpoint),
                "--onnx-opset",
                "13",
                "--ignore-unknown-parameters",
            ],
            cwd=tspp_main_dir,
            check=True,
        )
        # model-navigator run
        if cfg.inference.optimize:
            if cfg.inference.convert.type == "torchscript":
                with open(output_path + "/exported_model.pt.yaml", "r") as stream:
                    var_config = yaml.safe_load(stream)
                var_config_list = []
                for arg in ["--value-ranges", "--opt-shapes", "--dtypes"]:
                    var_config_list.append(arg)
                    if arg == "--value-ranges":
                        for k, v in var_config["inputs"].items():
                            var_config_list.append(k + "=0,0")
                    elif arg == "--opt-shapes":
                        for k, v in var_config["inputs"].items():
                            var_config_list.append(k + "=" + ",".join([str(x) for x in v["shape"]]))
                    else:
                        for k, v in var_config["inputs"].items():
                            var_config_list.append(k + "=" + v["dtype"])
            else:
                var_config_list = []
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
                    "{}".format(cfg.inference.convert.type),
                    "--model-format",
                    model_format,
                    "--triton-launch-mode",
                    "docker",
                    "--max-workspace-size",
                    "10000000000",
                    "--perf-measurement-request-count",
                    "100",
                    "--perf-analyzer-timeout",
                    "20",
                    "--concurrency",
                    "1",
                    "32",
                    "1024",
                    "--max-batch-size",
                    "{}".format(cfg.inference.batch_size),
                    "--gpus",
                    "all",
                    "--atol",
                    "1e-3",
                    "--rtol",
                    "100",
                    "--onnx-opsets",
                    "13",
                    "--container-version",
                    "21.09",
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
                    "{}".format(cfg.inference.convert.type),
                    "--model-format",
                    model_format,
                    "--launch-mode",
                    "local",
                    "--max-workspace-size",
                    "10000000000",
                    "--max-batch-size",
                    "{}".format(cfg.inference.batch_size),
                    "--target-precisions",
                    precision,
                    "--gpus",
                    "all",
                    "--atol",
                    "1e-3",
                    "--rtol",
                    "100",
                    "--onnx-opsets",
                    "13",
                    "--container-version",
                    "21.09",
                ],
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
                    "{}".format(cfg.inference.convert.type),
                    "--model-repository",
                    "{}/navigator_workspace/model-store/".format(output_path),
                    "--backend-accelerator",
                    cfg.inference.accelerator,
                    "--max-batch-size",
                    "{}".format(cfg.inference.batch_size),
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
                cfg.inference.convert.type if cfg.inference.convert.type != "torchscript" else cfg.inference.export.type
            )
            subprocess.run(
                [
                    "python",
                    "triton/check_accuracy.py",
                    "--native-model",
                    cfg.evaluator.checkpoint,
                    "--native-type",
                    "pyt",
                    "--export-model",
                    "{}/exported_model.pt".format(output_path),
                    "--export-type",
                    cfg.inference.export.type,
                    "--convert-model",
                    "{}/converted_model".format(output_path),
                    "--convert-type",
                    convert_type,
                    "--dataloader",
                    "triton/dataloader.py",
                    "--batch-size",
                    "{}".format(1),
                    "--model-dir",
                    "{}".format(cfg.evaluator.checkpoint),
                ],
                cwd=tspp_main_dir,
                check=True,
            )

    # get the actual model name
    if not os.path.isdir(os.path.join(output_path, "navigator_workspace")) or not os.path.isdir(
        os.path.join(output_path, "navigator_workspace/model-store")
    ):
        assert (
            False
        ), "This checkpoint directory is not configured correctly, there should be a dir/deployment/navigator_workspace/model-store/ directory"
    files_in_store = list(os.listdir(os.path.join(output_path, "navigator_workspace/model-store")))
    if len(files_in_store) < 1:
        assert False, "There needs to be exactly 1 model in the model-store directory"
    model_name = cfg.inference.get("model_name") if cfg.inference.get("model_name", None) else files_in_store[0]
    # deploy
    subprocess.run(["bash", "inference/deploy.sh", output_path, str(cfg.inference.gpu)], cwd=tspp_main_dir, check=True)
    # #create DL logger for this round of metrics
    # #run inference
    if not cfg.inference.just_deploy:
        os.makedirs(os.path.join(inference_dir, "raw"))
        dump_dir = os.path.join(inference_dir, "raw")
        dump_array = ["--dump-labels"]
        if config.config.evaluator.use_weights:
            dump_array.append("--dump-inputs")
        subprocess.run(
            [
                "python",
                "triton/run_inference_on_triton.py",
                "--model-name",
                model_name,
                "--model-version",
                "1",
                "--dataloader",
                "triton/dataloader.py",
                "--output-dir",
                "{}".format(dump_dir),
                "--batch-size",
                "{}".format(cfg.inference.batch_size),
                "--model-dir",
                "{}".format(cfg.evaluator.checkpoint),
            ]
            + dump_array,
            cwd=tspp_main_dir,
            check=True,
        )

        # calculate metrics
        subprocess.run(
            [
                "python",
                "triton/calculate_metrics.py",
                "--metrics",
                "triton/metrics.py",
                "--model-dir",
                "{}".format(cfg.evaluator.checkpoint),
                "--dump-dir",
                "{}".format(dump_dir),
                "--csv",
                "{}".format(os.path.join(inference_dir, "metrics.csv")),
            ],
            cwd=tspp_main_dir,
            check=True,
        )

        subprocess.run(["bash", "inference/stop_docker.sh"], cwd=tspp_main_dir)
