# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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


import os
import shutil
import subprocess
import hydra
from omegaconf import OmegaConf
from triton.dataloader import get_dataloader_fn

import dllogger
from data.data_utils import Preprocessor
from evaluators.evaluator import unpack_predictions

def run_inference_triton(config):
    cfg = config
    with open(os.path.join(cfg.checkpoint, ".hydra/config_merged.yaml"), "rb") as f:
        config = OmegaConf.load(f)
    config.evaluator = OmegaConf.merge(config.evaluator, cfg.evaluator)
    if cfg.get("dataset_dir", None):
        if not os.path.isdir(config.dataset.config.dest_path):
            raise ValueError("dataset_dir must be a directory")
        config.dataset.config.dest_path = cfg.dataset_dir
    config.inference = cfg
    with open(os.path.join(cfg.checkpoint, ".hydra/config_merged.yaml"), "wb") as f:
        OmegaConf.resolve(config)
        OmegaConf.save(config=config, f=f.name)
    output_path = os.path.join(cfg.checkpoint, "deployment")
    tspp_main_dir = os.path.sep + os.path.join(*(os.getcwd().split(os.path.sep)[:-3]))

    # get the actual model name
    if not os.path.isdir(os.path.join(output_path, "navigator_workspace")) or not os.path.isdir(
        os.path.join(output_path, "navigator_workspace/model-store")
    ):
        if os.path.isdir(os.path.join(output_path, "navigator_workspace/final-model-store")):
            shutil.copytree(os.path.join(output_path, "navigator_workspace/final-model-store"), os.path.join(output_path, "navigator_workspace/model-store"))
        else:
            assert (
                False
            ), "This checkpoint directory is not configured correctly, there should be a dir/deployment/navigator_workspace/model-store/ directory"
    files_in_store = list(os.listdir(os.path.join(output_path, "navigator_workspace/model-store")))
    if len(files_in_store) < 1:
        assert False, "There needs to be exactly 1 model in the model-store directory"

    evaluator = hydra.utils.call(config.evaluator)
    if config.dataset.config.get('xgb', False):
        if cfg.get("dataset_path", None):
            preprocessor = Preprocessor(config.dataset.config)
            if cfg.get("preproc_state_path", None):
                preprocessor_state_file = cfg.preproc_state_path
            else:
                preprocessor_state_file = None
            preprocessor.load_state(preprocessor_state_file)
            test_df = preprocessor.preprocess_test(dataset=cfg.dataset_path)
            test_df = preprocessor.apply_scalers(test_df)
            test_df = preprocessor.impute(test_df)
            train, valid, test = hydra.utils.call(config.dataset, input_df=test_df)
        else:
            train, valid, test = hydra.utils.call(config.dataset)
        del train, valid
        predictions_dict = evaluator.predict_xgboost(test, max_batch_size=cfg.batch_size)
        preds_full, labels_full, ids_full, weights_full, _ = unpack_predictions(predictions_dict)

    elif config.dataset.config.get('stat', False):
        raise ValueError("Stat models not supported on triton")
    else:
        model_name = cfg.get("model_name") if cfg.get("model_name", None) else files_in_store[0]
        dataloader = get_dataloader_fn(cfg.checkpoint, cfg.batch_size)
        predictions_dict = evaluator.predict(dataloader, model_name)
        preds_full, labels_full, ids_full, weights_full, _ = unpack_predictions(predictions_dict)

    metrics = evaluator.evaluate(preds_full, labels_full, ids_full, weights_full)
    logger = hydra.utils.call(config.logger)
    logger.log(step=[], data={k: float(v) for k, v in metrics.items()}, verbosity=dllogger.Verbosity.VERBOSE)
    logger.log(step='event', data={"String": "Evaluation Metrics: {}".format(metrics)}, verbosity=dllogger.Verbosity.DEFAULT)
    print(metrics)
