# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

class VolatilityConfig():
    def __init__(self):

        self.features = [
                         FeatureSpec('Symbol', InputTypes.ID, DataTypes.CATEGORICAL),
                         FeatureSpec('days_from_start', InputTypes.TIME, DataTypes.CONTINUOUS),
                         FeatureSpec('log_vol', InputTypes.TARGET, DataTypes.CONTINUOUS),
                         FeatureSpec('open_to_close', InputTypes.OBSERVED, DataTypes.CONTINUOUS),
                         FeatureSpec('days_from_start', InputTypes.KNOWN, DataTypes.CONTINUOUS),
                         FeatureSpec('day_of_week', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('day_of_month', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('week_of_year', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('month', InputTypes.KNOWN, DataTypes.CATEGORICAL),
                         FeatureSpec('Region', InputTypes.STATIC, DataTypes.CATEGORICAL),
                        ]

        # Dataset split boundaries
        self.time_ids = 'date' # This column contains time indices across which we split the data
        self.train_range = ('2000-01-01', '2016-01-01')
        self.valid_range = ('2016-01-01', '2018-01-01')
        self.test_range = ('2018-01-01', '2019-06-28')
        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = False
        self.missing_id_strategy = None
        self.missing_cat_data_strategy='encode_all'

        # Feature sizes
        self.static_categorical_inp_lens = [4]
        self.temporal_known_categorical_inp_lens = [7,31,53,12]
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 257
        self.encoder_length = 252

        self.n_head = 4
        self.hidden_size = 96
        self.dropout = 0.4
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

class FavoritaConfig():
    def __init__(self):
        self.features = [
            FeatureSpec('traj_id',         InputTypes.ID,       DataTypes.CATEGORICAL),
            #FeatureSpec('days_from_start', InputTypes.TIME,     DataTypes.CONTINUOUS),
            FeatureSpec('date',            InputTypes.TIME,     DataTypes.DATE),
            FeatureSpec('log_sales',       InputTypes.TARGET,   DataTypes.CONTINUOUS),
            # XXX for no apparent reason TF implementation doesn't scale day_of_month
            # and month variables. We probably should set them to be categorical
            FeatureSpec('day_of_month',    InputTypes.KNOWN,    DataTypes.CONTINUOUS),
            FeatureSpec('month',           InputTypes.KNOWN,    DataTypes.CONTINUOUS),
            FeatureSpec('onpromotion',     InputTypes.KNOWN,    DataTypes.CATEGORICAL),
            FeatureSpec('day_of_week',     InputTypes.KNOWN,    DataTypes.CATEGORICAL),
            FeatureSpec('national_hol',    InputTypes.KNOWN,    DataTypes.CATEGORICAL),
            FeatureSpec('regional_hol',    InputTypes.KNOWN,    DataTypes.CATEGORICAL),
            FeatureSpec('local_hol',       InputTypes.KNOWN,    DataTypes.CATEGORICAL),
            FeatureSpec('open',            InputTypes.KNOWN,    DataTypes.CONTINUOUS),
            FeatureSpec('transactions',    InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('oil',             InputTypes.OBSERVED, DataTypes.CONTINUOUS),
            FeatureSpec('categorical_id',  InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('item_nbr',        InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('store_nbr',       InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('city',            InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('state',           InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('type',            InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('cluster',         InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('family',          InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('class',           InputTypes.STATIC,   DataTypes.CATEGORICAL),
            FeatureSpec('perishable',      InputTypes.STATIC,   DataTypes.CATEGORICAL)
        ]

        # Dataset split boundaries
        self.time_ids = 'date' # This column contains time indices across which we split the data
        # When relative split is set then it is necessary to provide valid boundary.
        # Valid split is shifted from train split by number of forecast steps to the future
        # The test split is shifted by the number of forecast steps from the valid split
        self.relative_split = True
        self.valid_boundary = str(datetime.datetime(2015, 12, 1))

        self.train_range = None
        self.valid_range = None
        self.test_range = None

        self.dataset_stride = 1 #how many timesteps between examples
        self.scale_per_id = True
        self.missing_cat_data_strategy='encode_all'
        self.missing_id_strategy = 'drop'

        # Feature sizes
        self.static_categorical_inp_lens = [90200, 3426, 53, 22, 16, 5, 17, 32, 313, 2]
        self.temporal_known_categorical_inp_lens = [2, 7, 55, 5, 25]
        self.temporal_observed_categorical_inp_lens = []
        self.quantiles = [0.1, 0.5, 0.9]

        self.example_length = 120
        self.encoder_length = 90

        self.n_head = 4
        self.hidden_size = 240
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

CONFIGS = {'electricity':  ElectricityConfig,
           'volatility':   VolatilityConfig,
           'traffic':      TrafficConfig, 
           'favorita':     FavoritaConfig,
           }
