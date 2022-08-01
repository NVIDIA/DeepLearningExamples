# Copyright 2021-2022 NVIDIA Corporation

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
from typing import Union


class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""

    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3


DTYPE_MAP = {
    DataTypes.CONTINUOUS: np.float32,
    DataTypes.CATEGORICAL: np.int64,
    DataTypes.DATE: np.datetime64,
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


FEAT_ORDER = [
    (InputTypes.STATIC, DataTypes.CATEGORICAL),
    (InputTypes.STATIC, DataTypes.CONTINUOUS),
    (InputTypes.KNOWN, DataTypes.CATEGORICAL),
    (InputTypes.KNOWN, DataTypes.CONTINUOUS),
    (InputTypes.OBSERVED, DataTypes.CATEGORICAL),
    (InputTypes.OBSERVED, DataTypes.CONTINUOUS),
    (InputTypes.TARGET, DataTypes.CONTINUOUS),
    (InputTypes.WEIGHT, DataTypes.CONTINUOUS),
    (InputTypes.SAMPLE_WEIGHT, DataTypes.CONTINUOUS),
    (InputTypes.ID, DataTypes.CATEGORICAL),
]
FEAT_NAMES = ["s_cat", "s_cont", "k_cat", "k_cont", "o_cat", "o_cont", "target", "weight", "sample_weight", "id"]

def group_ids(df, features):
    col_names = ["_id_"] + [
                x.name
                for x in features
                if x.feature_embed_type != DataTypes.STR
                and x.feature_type != InputTypes.TIME
                and x.feature_type != InputTypes.ID
            ]
    grouped = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in df.groupby("_id_")]
    return grouped

def translate_features(features, preproc=False):
    all_features = [FeatureSpec(feature) for feature in features]
    if preproc:
        return all_features
    return [FeatureSpec({"name": "_id_", "feature_type": "ID", "feature_embed_type": "CATEGORICAL"})] + [
        feature for feature in all_features if feature.feature_type != InputTypes.ID
    ]


def map_dt(dt):
    if isinstance(dt, int):
        dt = dt
    elif isinstance(dt, ListConfig):
        dt = datetime.datetime(*dt)
    elif isinstance(dt, str):
        dt = datetime.datetime.strptime(dt, "%Y-%m-%d")
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


class Log1pScaler(FunctionTransformer):
    @staticmethod
    def _inverse(x):
        return np.exp(x) + 1

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
        for k, v in self.continuous_mapping.items():
            self.continuous_scalers[k] = {}
            if self.scale_per_id:
                for identifier, sliced in df.groupby("_id_"):
                    scaler = hydra.utils.instantiate(k).fit(sliced[v])
                    self.continuous_scalers[k][identifier] = scaler

            else:
                scaler = hydra.utils.instantiate(k).fit(df[v])
                self.continuous_scalers[k][""] = scaler

        for k, v in self.target_mapping.items():
            self.target_scalers[k] = {}
            if self.scale_per_id:
                for identifier, sliced in df.groupby("_id_"):
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
        # TODO: Assuming single targets for now. This has to be adapted to muti-target
        if len(self.target_scalers) > 0:

            shape = values.shape
            scalers = list(self.target_scalers.values())[0]
            if self.scale_per_id:
                assert ids is not None
                flat_values = values.flatten()
                flat_ids = np.repeat(ids, values.shape[1])
                df = pd.DataFrame({"id": flat_ids, "value": flat_values})
                df_list = []
                for identifier, sliced in df.groupby("id"):
                    df_list.append(np.stack(
                        [scalers[identifier].inverse_transform(sliced["value"].values.reshape(-1, 1)).flatten(),
                         sliced.index.values], axis=-1))
                tmp = np.concatenate(df_list)
                tmp = tmp[tmp[:, -1].argsort()]
                return tmp[:, 0].reshape(shape)
            else:
                flat_values = values.reshape(-1, 1)
                flat_values = scalers[""].inverse_transform(flat_values)
                return flat_values.reshape(shape)
        return values

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.features = translate_features(self.config["features"], preproc=True)
        self.feat_splits = self._get_feature_splits()
        self.cont_features_names = [continuous.name for continuous in self.feat_splits["input_continuous"]]
        self.dest_path = self.config.dest_path
        self.source_path = self.config.source_path
        self.preprocessor_state = {}
    
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

    def _get_dataset_splits(self, df):
        print("Splitting datasets")
        if hasattr(self.config, "valid_boundary") and self.config.valid_boundary is not None:
            forecast_len = self.config.example_length - self.config.encoder_length
            # The valid split is shifted from the train split by number of the forecast steps to the future.
            # The test split is shifted by the number of the forecast steps from the valid split
            valid_boundary = map_dt(self.config.valid_boundary)

            grouped = df.groupby('_id_')

            train_mask = grouped[self.config.time_ids].apply(lambda dates: dates < valid_boundary)
            train = df[train_mask]
            print('Calculated train.')
            train_sizes = train.groupby('_id_').size()

            valid_indexes = grouped[self.config.time_ids].apply(
                lambda dates: dates.iloc[(train_sizes[dates.name] - self.config.encoder_length):
                                        (train_sizes[dates.name] + forecast_len)].index
                              if dates.name in train_sizes else pd.Series()
                )
            valid = df.loc[np.concatenate(valid_indexes)]
            print('Calculated valid.')

            test_indexes = grouped[self.config.time_ids].apply(
                lambda dates: dates.iloc[(train_sizes[dates.name] - self.config.encoder_length + forecast_len):
                                        (train_sizes[dates.name] + 2 * forecast_len)].index
                              if dates.name in train_sizes else pd.Series()
                )
            test = df.loc[np.concatenate(test_indexes)]
            print('Calculated test.')
        elif df.dtypes[self.config.time_ids] not in [np.float64, np.int]:
            index = df[self.config.time_ids]

            train = df.loc[(index >= map_dt(self.config.train_range[0])) & (index < map_dt(self.config.train_range[1]))]
            valid = df.loc[(index >= map_dt(self.config.valid_range[0])) & (index < map_dt(self.config.valid_range[1]))]
            test = df.loc[(index >= map_dt(self.config.test_range[0])) & (index < map_dt(self.config.test_range[1]))]
        else:
            index = df[self.config.time_ids]
            train = df.loc[(index >= self.config.train_range[0]) & (index < self.config.train_range[1])]
            valid = df.loc[(index >= self.config.valid_range[0]) & (index < self.config.valid_range[1])]
            test = df.loc[(index >= self.config.test_range[0]) & (index < self.config.test_range[1])]

        train = train[(train.groupby('_id_').size()[train['_id_']] > self.config.encoder_length).values]
        valid = valid[(valid.groupby('_id_').size()[valid['_id_']] > self.config.encoder_length).values]
        test = test[(test.groupby('_id_').size()[test['_id_']] > self.config.encoder_length).values]

        return train, valid, test

    def _recombine_datasets(self, train, valid, test):
        if hasattr(self.config, "valid_boundary") and self.config.valid_boundary is not None:
            forecast_len = self.config.example_length - self.config.encoder_length
            # The valid split is shifted from the train split by number of the forecast steps to the future.
            # The test split is shifted by the number of the forecast steps from the valid split
            train_temp = []
            valid_temp = []
            for g0, g1 in zip(train.groupby("_id_"), valid.groupby("_id_")):
                _train = g0[1].iloc[: -self.config.encoder_length]
                _valid = g1[1].iloc[:forecast_len]
                train_temp.append(_train)
                valid_temp.append(_valid)
            train = pd.concat(train_temp, axis=0)
            valid = pd.concat(valid_temp, axis=0)
        elif train.dtypes[self.config.time_ids] not in [np.float64, np.int]:

            train = train[train[self.config.time_ids] < map_dt(self.config.valid_range[0])]
            valid = valid[valid[self.config.time_ids] < map_dt(self.config.test_range[0])]
        else:
            train = train[train[self.config.time_ids] < self.config.valid_range[0]]
            valid = valid[valid[self.config.time_ids] < self.config.test_range[0]]
        return pd.concat((train, valid, test))

    def _drop_unseen_categoricals(self, train, valid, test, drop_unseen=True):
        # TODO: Handle this for inference preprocess function
        if self.config.get("drop_unseen", False):
            print("Dropping unseen categoricals")
            if not drop_unseen:
                print("Warning: Assuming that inference dataset only has the input categoricals from the training set")
                return train, valid, test
            if hasattr(self.config, "valid_boundary") and self.config.valid_boundary is not None:
                arriter = ["_id_"]
            else:
                arriter = [cat.name for cat in self.feat_splits["input_categoricals"]] + ["_id_"]
            
            if train is not None:
                for categorical in arriter:
                    seen_values = train[categorical].unique()
                    valid = valid[valid[categorical].isin(seen_values)]
                    test = test[test[categorical].isin(seen_values)]
        return train, valid, test

    def fit_scalers(self, df):
        print("Calculating scalers")
        self.scaler = CompositeScaler(
            self.feat_splits["target_features"], self.feat_splits["input_continuous"], scale_per_id=self.config.get('scale_per_id', False)
        )
        self.scaler.fit(df)
        self.preprocessor_state["scalers"] = self.scaler

    def apply_scalers(self, df):
        print("Applying scalers")
        return self.preprocessor_state["scalers"].transform(df)

    def save_datasets(self, train, valid, test):
        print(F"Saving processed data at {self.dest_path}")
        os.makedirs(self.dest_path, exist_ok=True)

        train.to_csv(os.path.join(self.dest_path, "train.csv"))
        valid.to_csv(os.path.join(self.dest_path, "valid.csv"))
        test.to_csv(os.path.join(self.dest_path, "test.csv"))
        self._recombine_datasets(train, valid, test).to_csv(os.path.join(self.dest_path, "full.csv"))

        # Save relevant columns in binary form for faster dataloading
        # IMORTANT: We always expect id to be a single column indicating the complete timeseries
        # We also expect a copy of id in form of static categorical input!!!]]
        if self.config.get("binarized", False):
            grouped_train = group_ids(train, self.features)
            grouped_valid = group_ids(valid, self.features)
            grouped_test = group_ids(test, self.features)
            pickle.dump(grouped_train, open(os.path.join(self.dest_path, "train.bin"), "wb"))
            pickle.dump(grouped_valid, open(os.path.join(self.dest_path, "valid.bin"), "wb"))
            pickle.dump(grouped_test, open(os.path.join(self.dest_path, "test.bin"), "wb"))

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
            #TODO: check support for parse dates as done during read csv
            # Currently date related features are only used for dataset splits during training
            df = dataset.copy()
        else:
            raise ValueError(F"Function either accepts a path to a csv file or a dataframe")
        print("Sorting on time feature")
        #TODO: Check if we sort df for inference only case
        df = df.sort_values([self.feat_splits["time_feature"].name])
        f_names = [feature.name for feature in self.features] + [self.config.time_ids]
        df = df[list(dict.fromkeys(f_names))]
        
        if self.config.get("missing_data_label", False):
            df = df.replace(self.config.get("missing_data_label"), np.NaN)

        if drop_na:
            df = df.dropna(subset=[t.name for t in self.feat_splits["target_features"]])
        
        return df

    def preprocess(self):
        df = self._init_setup()
        self._map_ids(df)
        self._map_categoricals(df)
        train, valid, test = self._get_dataset_splits(df)
        train, valid, test = self._drop_unseen_categoricals(train, valid, test)
        return train, valid, test

    def preprocess_test(self, dataset: Union[str, pd.DataFrame]) -> pd.DataFrame:
        df = self._init_setup(dataset=dataset, drop_na=False)
        self._map_ids(df)
        self._map_categoricals(df)
        #TODO: this is a workaround and maybe needs to be handled properly in the future
        _, _, df = self._drop_unseen_categoricals(None, None, df, drop_unseen=False)
        return df