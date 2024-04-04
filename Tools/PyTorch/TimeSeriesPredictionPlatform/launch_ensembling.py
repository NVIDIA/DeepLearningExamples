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

"""Script assumes that ensembled models have the same output size (same data and model type).
Aimed to ensemble models from seed/hp sweeps
"""

import warnings
import os
import hydra
from omegaconf import OmegaConf

import conf.conf_utils # loads resolvers for OmegaConf
from evaluators.evaluator import unpack_predictions
from training.utils import set_seed
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="ensemble_conf")
def main(config):
    set_seed(config.get("seed", None))
    model_info = config.model.config.model_list[0]
    model_dir = model_info.dir
    with open(os.path.join(model_dir, '.hydra/config.yaml'), 'rb') as f:
        cfg = OmegaConf.load(f)

    if cfg.model._target_ == 'models.tspp_xgboost.TSPPXGBoost':
        config.model._target_ = 'models.ensembling.XGBEnsemble'

    model = hydra.utils.instantiate(config.model)

    train, valid, test = hydra.utils.call(cfg.dataset)
    del train, valid

    cfg.evaluator.config = {**cfg.evaluator.config, **config.evaluator.config}
    evaluator = hydra.utils.instantiate(cfg.evaluator, test_data=test)
    predictions_dict = evaluator.predict(model)
    preds, labels, ids, weights, timestamps, _ = unpack_predictions(predictions_dict)
    eval_metrics = evaluator.evaluate(preds, labels, ids, weights, timestamps)
    logger = hydra.utils.call(config.logger)
    logger.log(step=[], data=eval_metrics, verbosity=0)
    logger.flush()

if __name__ == '__main__':
    main()
