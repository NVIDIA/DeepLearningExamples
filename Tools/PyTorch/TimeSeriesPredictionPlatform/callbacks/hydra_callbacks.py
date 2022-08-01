# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import pandas as pd

from omegaconf import OmegaConf
from hydra.experimental.callback import Callback

from loggers.log_helper import jsonlog_2_df

class MergeLogs(Callback):
    def on_multirun_end(self, config, **kwargs):
        OmegaConf.resolve(config)

        ALLOWED_KEYS=['timestamp', 'elapsed_time', 'step', 'loss', 'val_loss', 'MAE', 'MSE', 'RMSE', 'P50', 'P90']

        dfs = []
        for p, sub_dirs, files in os.walk(config.hydra.sweep.dir):
            if 'log.json' in files:
                path = os.path.join(p, 'log.json')
                df = jsonlog_2_df(path, ALLOWED_KEYS)
                dfs.append(df)

        # Transpose dataframes
        plots = {}
        for c in dfs[0].columns:
            joint_plots = pd.DataFrame({i : df[c] for i, df in enumerate(dfs)})
            metrics = {}
            metrics['mean'] = joint_plots.mean(axis=1)
            metrics['std'] = joint_plots.std(axis=1)
            metrics['mean_m_std'] = metrics['mean'] - metrics['std']
            metrics['mean_p_std'] = metrics['mean'] + metrics['std']
            metrics_df = pd.DataFrame(metrics)
            plots[c] = metrics_df[~metrics_df.isna().all(axis=1)] # Drop rows which contain only NaNs

        timestamps = plots.pop('timestamp')['mean']
        timestamps = (timestamps * 1000).astype(int)
        if not timestamps.is_monotonic:
            raise ValueError('Timestamps are not monotonic')
