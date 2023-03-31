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

from data_utils import InputTypes, DataTypes, FeatureSpec
import datetime

class ElectricityConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('power_usage', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('hour', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'days_from_start' # This column contains time indices across which we split the data
        self.train_range = (1096, 1315)
        self.valid_range = (1308, 1339)
        self.test_range = (1332, 1346)
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [369]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.1
        self.attn_dropout = 0.0

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


class TrafficConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('id', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('hours_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('values', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('time_on_day', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('hours_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('categorical_id', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]
        # Dataset split boundaries
        self.time_ids = 'sensor_day' # This column contains time indices across which we split the data
        self.train_range = (0, 151)
        self.valid_range = (144, 166)
        self.test_range = (159, float('inf'))
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = False
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [963]
        self.temporal_known_categorical_inp_lens = []
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 8 * 24
        self.encoder_length = 7 * 24

        self.n_head = 4
        self.hidden_size = 128
        self.dropout = 0.3
        self.attn_dropout = 0.0

        #### Derived variables ####
        self.temporal_known_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.KNOWN and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_observed_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.OBSERVED and x.feature_embed_type == DataTypes.CONTINUOUS])
        self.temporal_target_size = len([x for x in self.features if x.feature_type == InputTypes.TARGET])
        self.static_continuous_inp_size = len([x for x in self.features 
            if x.feature_type == InputTypes.STATIC and x.feature_embed_type == DataTypes.CONTINUOUS])

        self.num_static_vars = self.static_continuous_inp_size + len(self.static_categorical_inp_lens)
        self.num_future_vars = self.temporal_known_continuous_inp_size + len(self.temporal_known_categorical_inp_lens)
        self.num_historic_vars = sum([self.num_future_vars,
                                      self.temporal_observed_continuous_inp_size,
                                      self.temporal_target_size,
                                      len(self.temporal_observed_categorical_inp_lens),
                                      ])


CONFIGS = {'electricity':  ElectricityConfig,
           'traffic':      TrafficConfig,
           }
