# Copyright 2021-2024 NVIDIA Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2020 The Google Research Authors.
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

import datetime
import enum
import os
import pickle

import hydra
import numpy as np
import pandas as pd
from omegaconf.listconfig import ListConfig
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from typing import Union, List, Dict


class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""

    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3


DTYPE_MAP = {
    DataTypes.CONTINUOUS: np.float32,
    DataTypes.CATEGORICAL: np.int64,
    DataTypes.DATE: 'datetime64[ns]', #This can be a string because this is meant to be used as an argument to ndarray.astype 
    DataTypes.STR: str,
}


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""

    TARGET = 0
    OBSERVED = 1
    KNOWN = 2
    STATIC = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index
    WEIGHT = 6
    SAMPLE_WEIGHT = 7


class FeatureSpec:
    enabled_attributes = ["name", "feature_type", "feature_embed_type", "cardinality", "scaler"]

    def __init__(self, input_dict):
        for key in input_dict:
            if key in self.enabled_attributes:
                setattr(self, key, input_dict[key])
            else:
                raise ValueError("Attribute not enabled: {attr}".format(attr=key))
        self.name = input_dict["name"]
        self.feature_type = InputTypes[input_dict["feature_type"]]
        self.feature_embed_type = DataTypes[input_dict["feature_embed_type"]]

    def get(self, key, value=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return value

    def __str__(self):
        return str((self.name, self.feature_type, self.feature_embed_type))

    def __repr__(self):
        return str(self)

# Since Python 3.7, dictionaries are ordered by default and maintain their insertion order
FEAT_NAME_MAP = {
    "s_cat":             (InputTypes.STATIC, DataTypes.CATEGORICAL),
    "s_cont":            (InputTypes.STATIC, DataTypes.CONTINUOUS),
    "k_cat":             (InputTypes.KNOWN, DataTypes.CATEGORICAL),
    "k_cont":            (InputTypes.KNOWN, DataTypes.CONTINUOUS),
    "o_cat":             (InputTypes.OBSERVED, DataTypes.CATEGORICAL),
    "o_cont":            (InputTypes.OBSERVED, DataTypes.CONTINUOUS),
    "target":            (InputTypes.TARGET, DataTypes.CONTINUOUS),
    "weight":            (InputTypes.WEIGHT, DataTypes.CONTINUOUS),
    "sample_weight":     (InputTypes.SAMPLE_WEIGHT, DataTypes.CONTINUOUS),
    "id":                (InputTypes.ID, DataTypes.CATEGORICAL),
    "timestamp":         (InputTypes.TIME, DataTypes.CATEGORICAL) # During preprocessing we cast all time data to int
}


def group_ids(df, features):
    sizes = df['_id_'].value_counts(dropna=False, sort=False).sort_index()
    #sizes = sizes[sizes >= example_length]
    #valid_ids = set(sizes.index)
    sizes = sizes.values
    
    feature_col_map = {k: [
                f.name for f in features 
                if (f.feature_type, f.feature_embed_type) == v
            ]
            for k, v in FEAT_NAME_MAP.items()
        }

    # These 2 columns are defined at preprocessing stage. We should redesign it so it wouldn't be necessary
    feature_col_map['id'] = ['_id_']
    feature_col_map['timestamp'] = ['_timestamp_']
    
    # df is sorted by _id_ and time feature, so there is no need to group df
    grouped = [
        df.loc[:, feature_col_map[feat]].values.astype(DTYPE_MAP[dtype])
        for feat, (_, dtype) in FEAT_NAME_MAP.items()
    ]
    return grouped, sizes


def translate_features(features, preproc=False):
    all_features = [FeatureSpec(feature) for feature in features]
    if preproc:
        return all_features
    return [FeatureSpec({"name": "_id_", "feature_type": "ID", "feature_embed_type": "CATEGORICAL"}),
            FeatureSpec({"name": "_timestamp_", "feature_type": "TIME", "feature_embed_type": "CATEGORICAL"})] + \
            [feature for feature in all_features if feature.feature_type not in [InputTypes.ID, InputTypes.TIME]]


def map_dt(dt):
    if isinstance(dt, int):
        dt = dt
    elif isinstance(dt, ListConfig):
        dt = datetime.datetime(*dt)
    elif isinstance(dt, str):
        try:
            dt = datetime.datetime.strptime(dt, "%Y-%m-%d")
        except ValueError:
            dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')

    return dt


def impute(df, config):
    if not (config.get("missing_data_label", False)):
        return df, None
    else:
        imp = SimpleImputer(missing_values=config.missing_data_label, strategy="mean")
        mask = df.applymap(lambda x: True if x == config.missing_data_label else False)
        data = df.values
        col_mask = (data == config.missing_data_label).all(axis=0)
        data[:, ~col_mask] = imp.fit_transform(data)
        return data, mask


def map_scalers(features):
    mapping = {}
    for feature in features:
        if feature.get("scaler", None):
            if mapping.get(feature.scaler, None):
                mapping[feature.scaler].append(feature.name)
            else:
                mapping[feature.scaler] = [feature.name]
    return mapping


def get_alignment_compliment_bytes(size, dtype):
    # return two's compliment for the dtype, so new array starts on multiple of dtype 
    return (~size + 1) & (dtype.alignment - 1)  

class Log1pScaler(FunctionTransformer):
    @staticmethod
    def _inverse(x):
        return np.expm1(x)

    def __init__(self):
        super().__init__(func=np.log1p, inverse_func=Log1pScaler._inverse, validate=False)


class CompositeScaler:
    def __init__(self, target_features, input_continuous, scale_per_id):
        self.target_mapping = map_scalers(target_features)
        self.continuous_mapping = map_scalers(input_continuous)
        self.target_features = target_features
        self.input_continuous = input_continuous
        self.scale_per_id = scale_per_id
        self.continuous_scalers = {}
        self.target_scalers = {}

    def fit(self, df):
        if self.scale_per_id:
            grouped_df = df.groupby("_id_")

        for k, v in self.continuous_mapping.items():
            self.continuous_scalers[k] = {}
            if self.scale_per_id:
                for identifier, sliced in grouped_df:
                    scaler = hydra.utils.instantiate(k).fit(sliced[v])
                    self.continuous_scalers[k][identifier] = scaler

            else:
                scaler = hydra.utils.instantiate(k).fit(df[v])
                self.continuous_scalers[k][""] = scaler

        for k, v in self.target_mapping.items():
            self.target_scalers[k] = {}
            if self.scale_per_id:
                for identifier, sliced in grouped_df:
                    scaler = hydra.utils.instantiate(k).fit(sliced[v])
                    self.target_scalers[k][identifier] = scaler

            else:
                scaler = hydra.utils.instantiate(k).fit(df[v])
                self.target_scalers[k][""] = scaler

    def apply_scalers(self, df, name=None):
        if name is None:
            name = df.name
        for k, v in self.continuous_mapping.items():
            df[v] = self.continuous_scalers[k][name].transform(df[v])
        for k, v in self.target_mapping.items():
            df[v] = self.target_scalers[k][name].transform(df[v])
        return df

    def transform(self, df):
        if self.scale_per_id:
            df = df.groupby("_id_").apply(self.apply_scalers)
        else:
            df = self.apply_scalers(df, name="")
        return df
    
    def inverse_transform_targets(self, values, ids=None):
        if len(self.target_scalers) <= 0:
            return values
        scalers = list(self.target_scalers.values())[0]

        # Assumption in 4D case: ids: NxI, values: NxTxIxH
        if self.scale_per_id:
            assert ids is not None
            # Move time id to the second dim
            if len(values.shape) == 4:
                values = values.transpose(0,2,1,3)

            uids = np.unique(ids)
            inversed = np.zeros_like(values)
            for i in uids:
                idx = ids == i
                x = values[idx]
                x = scalers[i].inverse_transform(x)
                inversed[idx] = x

            if len(values.shape) == 4:
                inversed = inversed.transpose(0,2,1,3)
            return inversed
        else:
            flat_values = values.reshape(-1, 1)
            flat_values = scalers[""].inverse_transform(flat_values)
            return flat_values.reshape(values.shape)

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.features = translate_features(self.config["features"], preproc=True)
        self.feat_splits = self._get_feature_splits()
        self.cont_features_names = [continuous.name for continuous in self.feat_splits["input_continuous"]]
        self.dest_path = self.config.dest_path
        self.source_path = self.config.source_path
        self.preprocessor_state = {}
        self.scaler = None
        self.alt_scaler = None
    
    def _get_feature_splits(self):
        splits = {}
        splits["dates"] = [feature for feature in self.features if feature.feature_embed_type == DataTypes.DATE]
        splits["target_features"] = [feature for feature in self.features if feature.feature_type == InputTypes.TARGET]
        splits["time_feature"] = [feature for feature in self.features if feature.feature_type == InputTypes.TIME][0]
        splits["id_features"] = [feature for feature in self.features if feature.feature_type == InputTypes.ID]
        splits["input_categoricals"] = [
            feature
            for feature in self.features
            if feature.feature_embed_type == DataTypes.CATEGORICAL
            and feature.feature_type in [InputTypes.STATIC, InputTypes.KNOWN, InputTypes.OBSERVED]
        ]
        splits["input_continuous"] = [
            feature
            for feature in self.features
            if feature.feature_embed_type == DataTypes.CONTINUOUS
            and feature.feature_type in [InputTypes.STATIC, InputTypes.KNOWN, InputTypes.OBSERVED]
        ]
        return splits
    
    def _map_ids(self, df):
        print("Mapping nodes")
        id_features = [feature.name for feature in self.feat_splits["id_features"]]
        if "id_mappings" in self.preprocessor_state:
            id_features_df = self.preprocessor_state["id_mappings"]
            id_features_dict = id_features_df.set_index(id_features).to_dict()["_id_"]
            def id_map_funct(x):
                var = tuple(x[id_features])
                if len(var) == 1:
                    var = var[0]
                return id_features_dict.get(var, np.nan)
            df["_id_"] = df.apply(lambda x: id_map_funct(x), axis=1)
        else:
            id_features = [feature.name for feature in self.feat_splits["id_features"]]
            current_id = df[id_features[0]].astype("category").cat.codes + 1
            for additional_id in id_features[1:]:
                current_id = df[additional_id].astype("category").cat.codes * (current_id.max() + 1) + current_id + 1
            df["_id_"] = current_id.astype("category").cat.codes
            id_features_df = df[id_features + ["_id_"]]
            id_features_df = id_features_df.drop_duplicates(subset=None).reset_index(drop=True)
            self.preprocessor_state["id_mappings"] = id_features_df              

    def _map_categoricals(self, df):
        print("Mapping categoricals to bounded range")
        if "categorical_mappings" in self.preprocessor_state:
            categorical_mappings = self.preprocessor_state["categorical_mappings"]
            for categorical in self.feat_splits['input_categoricals']:
                df[categorical.name] = df[categorical.name].map(categorical_mappings[categorical.name])
        else:
            input_categorical_map_dict = {}
            for categorical in self.feat_splits['input_categoricals']:
                cat_feature = df[categorical.name].astype("category")
                input_categorical_map_dict[categorical.name] = dict(zip([np.nan] + cat_feature.cat.categories.tolist(), 
                                                                        range(0, len(cat_feature.cat.categories)+1)))
                df[categorical.name] = cat_feature.cat.codes + 1
            self.preprocessor_state["categorical_mappings"] = input_categorical_map_dict

    def _map_time_col(self, df):
        time_feat = self.feat_splits["time_feature"].name
        df['_timestamp_'] = df[time_feat]
        self.preprocessor_state['timestamp_embed_type'] = self.feat_splits["time_feature"].feature_embed_type

    def _get_dataset_splits(self, df):
        print("Splitting datasets")
        time_feat = self.feat_splits['time_feature']
        if hasattr(self.config, "valid_boundary") and self.config.valid_boundary is not None:
            forecast_len = self.config.example_length - self.config.encoder_length
            # The valid split is shifted from the train split by number of the forecast steps to the future.
            # The test split is shifted by the number of the forecast steps from the valid split
            valid_boundary = map_dt(self.config.valid_boundary)

            grouped = df.groupby('_id_')

            train_mask = grouped[time_feat.name].apply(lambda dates: dates < valid_boundary)
            train = df[train_mask]
            print('Calculated train.')
            train_sizes = train.groupby('_id_').size()
            exclude_name = train_sizes < self.config.example_length

            valid_indexes = grouped[time_feat.name].apply(
                lambda dates: dates.iloc[(train_sizes[dates.name] - self.config.encoder_length):
                                        (train_sizes[dates.name] + forecast_len)].index
                              if dates.name in train_sizes and not exclude_name[dates.name] else pd.Series()
                )
            valid = df.loc[np.concatenate(valid_indexes)]
            print('Calculated valid.')

            test_indexes = grouped[time_feat.name].apply(
                lambda dates: dates.iloc[(train_sizes[dates.name] - self.config.encoder_length + forecast_len):
                                        (train_sizes[dates.name] + 2 * forecast_len)].index
                              if dates.name in train_sizes and not exclude_name[dates.name] else pd.Series()
                )
            test = df.loc[np.concatenate(test_indexes)]
            print('Calculated test.')
        elif time_feat.feature_embed_type == DataTypes.DATE:
            index = df[time_feat.name]

            train = df.loc[(index >= map_dt(self.config.train_range[0])) & (index < map_dt(self.config.train_range[1]))]
            valid = df.loc[(index >= map_dt(self.config.valid_range[0])) & (index < map_dt(self.config.valid_range[1]))]
            test = df.loc[(index >= map_dt(self.config.test_range[0])) & (index < map_dt(self.config.test_range[1]))]
        else:
            index = df[time_feat.name]

            train = df.loc[(index >= self.config.train_range[0]) & (index < self.config.train_range[1])]
            valid = df.loc[(index >= self.config.valid_range[0]) & (index < self.config.valid_range[1])]
            test = df.loc[(index >= self.config.test_range[0]) & (index < self.config.test_range[1])]

        train = train[(train.groupby('_id_').size()[train['_id_']] >= self.config.example_length).values]
        valid = valid[(valid.groupby('_id_').size()[valid['_id_']] >= self.config.example_length).values]
        test = test[(test.groupby('_id_').size()[test['_id_']] >= self.config.example_length).values]

        return train, valid, test
    
    def _get_dataset_splits_stat(self, df):
        print("Splitting stats datasets")
        time_feat = self.feat_splits['time_feature']
        forecast_len = self.config.example_length - self.config.encoder_length
        if hasattr(self.config, "valid_boundary") and self.config.valid_boundary is not None:
            # The valid split is shifted from the train split by number of the forecast steps to the future.
            # The test split is shifted by the number of the forecast steps from the valid split
            valid_boundary = map_dt(self.config.valid_boundary)

            data_sizes = df['_id_'].value_counts(dropna=False, sort=False)
            train_sizes = df.loc[df[time_feat.name] < valid_boundary, '_id_'].value_counts(dropna=False, sort=False)
            exclude_name = train_sizes < self.config.example_length

            grouped = df.groupby('_id_')

            train_stat_index = grouped[time_feat.name].apply(
                lambda dates: dates.iloc[:train_sizes[dates.name] + forecast_len].index
                       if dates.name in train_sizes and not exclude_name[dates.name] and train_sizes[dates.name] + 2*forecast_len <= data_sizes[dates.name] else pd.Series()
            )
            train_stat = df.loc[np.concatenate(train_stat_index)]
            print('Calculated stat train.')
            test_stat_indexes = grouped[time_feat.name].apply(
                lambda dates: dates.iloc[train_sizes[dates.name] + forecast_len:
                                        train_sizes[dates.name] + 2*forecast_len].index
                              if dates.name in train_sizes and not exclude_name[dates.name] and train_sizes[dates.name] + 2*forecast_len <= data_sizes[dates.name] else pd.Series()
                )
            test_stat = df.loc[np.concatenate(test_stat_indexes)]
            print('Calculated stat test.')
            return train_stat, test_stat
        elif time_feat.feature_embed_type == DataTypes.DATE:
            index = df[time_feat.name]

            delta = (index[1] - index[0]) * self.config.encoder_length

            train_stat = df.loc[(index >= map_dt(self.config.train_range[0])) & (index < map_dt(self.config.test_range[0]) + delta)]
            test_stat = df.loc[(index >= map_dt(self.config.test_range[0]) + delta) & (index < map_dt(self.config.test_range[1]))]
        else:
            index = df[time_feat.name]
            train_stat = df.loc[(index >= self.config.train_range[0]) & (index < self.config.test_range[0] + self.config.encoder_length)]
            test_stat = df.loc[(index >= self.config.test_range[0] + self.config.encoder_length) & (index < self.config.test_range[1])]
        
        train_sizes = train_stat['_id_'].value_counts(dropna=False, sort=False)
        test_sizes = test_stat['_id_'].value_counts(dropna=False, sort=False)

        # filter short examples
        train_sizes = train_sizes[train_sizes >= self.config.example_length]
        test_sizes = test_sizes[test_sizes >= forecast_len]

        # cross check sets to ensure that train and test contain the same _id_'s
        train_sizes = train_sizes[train_sizes.index.isin(test_sizes.index)]
        test_sizes = test_sizes[test_sizes.index.isin(train_sizes.index)]

        train_stat[train_stat['_id_'].isin(train_sizes.index)]
        test_stat[test_stat['_id_'].isin(test_sizes.index)]
        return train_stat, test_stat

    def _drop_unseen_categoricals(self, train, valid=None, test=None, drop_unseen=True):
        if self.config.get("drop_unseen", False):
            print("Dropping unseen categoricals")
            if not drop_unseen:
                print("Warning: Assuming that inference dataset only has the input categoricals from the training set")
                return train, valid, test
            if hasattr(self.config, "valid_boundary") and self.config.valid_boundary is not None:
                arriter = ["_id_"]
            else:
                arriter = [cat.name for cat in self.feat_splits["input_categoricals"]] + ["_id_"]
            
            if train is not None and (valid is not None or test is not None):
                for categorical in arriter:
                    seen_values = train[categorical].unique()
                    if valid is not None:
                        valid = valid[valid[categorical].isin(seen_values)]
                    if test is not None:
                        test = test[test[categorical].isin(seen_values)]
        return train, valid, test

    def fit_scalers(self, df, alt_scaler=False):
        print("Calculating scalers")
        scaler = CompositeScaler(
            self.feat_splits["target_features"], self.feat_splits["input_continuous"], scale_per_id=self.config.get('scale_per_id', False)
        )
        scaler.fit(df)
        if alt_scaler:
            self.alt_scaler = scaler
            self.preprocessor_state["alt_scalers"] = scaler
        else:
            self.scaler = scaler
            self.preprocessor_state["scalers"] = scaler

    def apply_scalers(self, df, alt_scaler=False):
        print("Applying scalers")
        return self.preprocessor_state["alt_scalers" if alt_scaler else "scalers"].transform(df)

    def save_datasets(self, train, valid, test, train_stat, test_stat):
        print(F"Saving processed data at {self.dest_path}")
        os.makedirs(self.dest_path, exist_ok=True)

        train.to_csv(os.path.join(self.dest_path, "train.csv"))
        valid.to_csv(os.path.join(self.dest_path, "valid.csv"))
        test.to_csv(os.path.join(self.dest_path, "test.csv"))
        train_stat.to_csv(os.path.join(self.dest_path, "train_stat.csv"))
        test_stat.to_csv(os.path.join(self.dest_path, "test_stat.csv"))

        # Save relevant columns in binary form for faster dataloading
        # IMORTANT: We always expect id to be a single column indicating the complete timeseries
        # We also expect a copy of id in form of static categorical input!!!]]
        if self.config.get("binarized", False):
            train = group_ids(train, self.features)
            valid = group_ids(valid, self.features)
            test = group_ids(test, self.features)
            train_stat = group_ids(train_stat, self.features)
            test_stat = group_ids(test_stat, self.features)

            for file, (grouped_ds, sizes) in (('train.bin', train),
                                              ('valid.bin', valid),
                                              ('test.bin', test),
                                              ('train_stat.bin', train_stat),
                                              ('test_stat.bin', test_stat)):
                metadata = {
                    'group_sizes': sizes,
                    'col_desc': [
                        (g.dtype, g.shape, g.nbytes) for g in grouped_ds
                    ]
                }
                with open(os.path.join(self.dest_path, file), "wb") as f:
                    pickle.dump(metadata, f)
                    for col in grouped_ds:
                        f.write(b'\0' * get_alignment_compliment_bytes(f.tell(), col.dtype))
                        col.tofile(f)
                    

    def save_state(self):
        filepath = os.path.join(self.dest_path, "tspp_preprocess.bin")
        print(F"Saving preprocessor state at {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self.preprocessor_state, f)

    def load_state(self, preprocessor_state_file):
        filepath = os.path.join(self.config.dest_path, "tspp_preprocess.bin")
        if preprocessor_state_file:
            filepath = preprocessor_state_file
        if not os.path.exists(filepath):
            raise ValueError(F"Invalid preprocessor state file: {filepath}")

        print(F"Reading preprocessor state binary file: {filepath}")
        f = open(filepath, "rb")
        self.preprocessor_state = pickle.load(f)
        required_keys = ("id_mappings", "categorical_mappings", "scalers")
        if not all(k in self.preprocessor_state for k in required_keys):
            raise ValueError(F"preprocessor state binary file at :{filepath} must have keys={required_keys} but found={self.preprocessor_state.keys()}")

    def impute(self, df):
        print("Fixing any nans in continuous features")
        df[self.cont_features_names] = df[self.cont_features_names].replace(np.NaN, 10 ** 9)
        return df

    def _init_setup(self, dataset=None, drop_na=True):
        if dataset is None:
            print(F"Reading in data from CSV File: {self.source_path}")
            df = pd.read_csv(self.source_path, parse_dates=[d.name for d in self.feat_splits["dates"]])
        elif isinstance(dataset, str) and dataset.endswith(".csv"):
            print(F"Reading in data from CSV File: {dataset}")
            df = pd.read_csv(dataset, parse_dates=[d.name for d in self.feat_splits["dates"]])
        elif isinstance(dataset, pd.DataFrame):
            print("Input DataFrame provided for preprocessing")
            # Currently date related features are only used for dataset splits during training
            df = dataset.copy()
        else:
            raise ValueError(F"Function either accepts a path to a csv file or a dataframe")

        print("Sorting on time feature")
        df = df.sort_values([id_feat.name for id_feat in self.feat_splits["id_features"]] + [self.feat_splits["time_feature"].name])
        f_names = {feature.name for feature in self.features}
        df = df[f_names]
        
        if self.config.get("missing_data_label", False):
            df = df.replace(self.config.get("missing_data_label"), np.NaN)

        if drop_na:
            df = df.dropna(subset=[t.name for t in self.feat_splits["target_features"]])
        
        return df

    def preprocess(self):
        df = self._init_setup()
        self._map_ids(df)
        self._map_time_col(df)
        self._map_categoricals(df)
        train, valid, test = self._get_dataset_splits(df)
        train, valid, test = self._drop_unseen_categoricals(train, valid, test)

        train_stat, test_stat = self._get_dataset_splits_stat(df)
        train_stat, _, test_stat = self._drop_unseen_categoricals(train_stat, test=test_stat)
        return train, valid, test, train_stat, test_stat

    def preprocess_test(self, dataset: Union[str, pd.DataFrame]) -> pd.DataFrame:
        df = self._init_setup(dataset=dataset, drop_na=False)
        self._map_ids(df)
        self._map_categoricals(df)
        _, _, df = self._drop_unseen_categoricals(None, None, df, drop_unseen=False)
        return df
