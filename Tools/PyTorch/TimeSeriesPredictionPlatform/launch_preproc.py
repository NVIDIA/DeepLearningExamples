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

# SPDX-License-Identifier: Apache-2.0

import warnings

import hydra

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf/", config_name="preproc_config")
def main(cfg):
    print(cfg)
    preprocessor = hydra.utils.instantiate(cfg, _recursive_=False)
    train, valid, test, train_stat, test_stat = preprocessor.preprocess()

    preprocessor.fit_scalers(train)
    preprocessor.fit_scalers(train_stat, alt_scaler=True)

    train = preprocessor.apply_scalers(train)
    valid = preprocessor.apply_scalers(valid)
    test = preprocessor.apply_scalers(test)

    train_stat = preprocessor.apply_scalers(train_stat, alt_scaler=True)
    test_stat = preprocessor.apply_scalers(test_stat, alt_scaler=True)

    train = preprocessor.impute(train)
    valid = preprocessor.impute(valid)
    test = preprocessor.impute(test)

    train_stat = preprocessor.impute(train_stat)
    test_stat = preprocessor.impute(test_stat)

    preprocessor.save_state()
    preprocessor.save_datasets(train, valid, test, train_stat, test_stat)

if __name__ == "__main__":
    main()
