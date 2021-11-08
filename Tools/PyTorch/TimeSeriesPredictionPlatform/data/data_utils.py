# Copyright 2021 NVIDIA CORPORATION

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
import math
import os
import pickle
from abc import ABC
from bisect import bisect
from collections import namedtuple
from itertools import combinations

import hydra
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from dgl.transform import metis_partition_assignment
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset


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


def translate_features(features, preproc=False):
    all_features = [FeatureSpec(feature) for feature in features]
    if preproc:
        return all_features
    return [FeatureSpec({"name": "_id_", "feature_type": "ID", "feature_embed_type": "CATEGORICAL"})] + [
        feature for feature in all_features if feature.feature_type != InputTypes.ID
    ]


class TSBaseDataset(Dataset):
    def __init__(self, features, path=None, encoder_length=52, example_length=54, stride=1):
        super().__init__()
        assert example_length > encoder_length
        self.features = features
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.stride = stride
        self.path = path
        self.load()
        self.features = [i for i in self.features if i.feature_type != InputTypes.TIME]
        self.grouped = [x for x in self.grouped if x.shape[0] >= self.example_length]
        self.group_lens = [(g.shape[0] - self.example_length + 1) // self.stride for g in self.grouped]
        self._cum_examples_in_group = np.cumsum(self.group_lens)

        self.feature_type_col_map = [
            [i for i, f in enumerate(self.features) if (f.feature_type, f.feature_embed_type) == x] for x in FEAT_ORDER
        ]

        self.grouped = [
            [
                arr[:, idxs].view(dtype=np.float32).astype(DTYPE_MAP[t[1]])
                for t, idxs in zip(FEAT_ORDER, self.feature_type_col_map)
            ]
            for arr in self.grouped
        ]

    def get_probabilities(self):
        sampled = []
        for i in range(len(self.grouped)):
            group_len = self.group_lens[i]
            group = self.grouped[i]
            sample_weights = group[-1]
            sampled.append(sample_weights[np.arange(0, self.stride * group_len, self.stride)])
        sampled = np.concatenate(sampled)
        return sampled

    def __len__(self):
        return self._cum_examples_in_group[-1]

    def __getitem__(self, idx):
        g_idx = bisect(self._cum_examples_in_group, idx)
        e_idx = idx - self._cum_examples_in_group[g_idx - 1] if g_idx else idx

        group = self.grouped[g_idx]

        tensors = [
            torch.from_numpy(feat[e_idx * self.stride : e_idx * self.stride + self.example_length])
            if feat.size
            else torch.empty(0)
            for feat in group
        ]

        out = dict(zip(FEAT_NAMES, tensors))

        # XXX: dataset shouldn't be aware of encoder_lenght probably. Masking should occur on some other level
        out["weight"] = out["weight"][self.encoder_length :, :] if out["weight"].numel() else out["weight"]
        out["id"] = out["id"][0, :]
        return out


class TSDataset(TSBaseDataset):
    def load(self):
        data = pd.read_csv(self.path)
        col_names = ["_id_"] + [
            x.name
            for x in self.features
            if x.feature_embed_type != DataTypes.STR
            and x.feature_type != InputTypes.TIME
            and x.feature_type != InputTypes.ID
        ]

        self.grouped = [group[1][col_names].values.astype(np.float32).view(dtype=np.int32) for group in data.groupby("_id_")]


class TSBinaryDataset(TSBaseDataset):
    def load(self):
        self.grouped = pickle.load(open(self.path, "rb"))


class StatDataset(Dataset):
    def __init__(self, features, path=None, encoder_length=52, example_length=54, stride=1, split=None, split_feature=None):
        super().__init__()
        assert example_length > encoder_length
        self.features = translate_features(features)
        self.time_feature = split_feature
        self.weight_features = [feature.name for feature in self.features if feature.feature_type == InputTypes.WEIGHT]
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.horizon = self.example_length - self.encoder_length
        self.stride = stride
        self.split = split

        self.id_col_name = next(x.name for x in self.features if x.feature_type == InputTypes.ID)
        self.col_dtypes = {v.name: DTYPE_MAP[v.feature_embed_type] for v in self.features}
        self.data = pd.read_csv(os.path.join(path, "full.csv"), dtype=self.col_dtypes)
        self.data = self.data.groupby(self.id_col_name).filter(lambda group: len(group) >= self.example_length)
        self.grouped = list(self.data.groupby(self.id_col_name))
        self.endog = [feature.name for feature in self.features if feature.feature_type == InputTypes.TARGET]
        self.exog = [
            feature.name
            for feature in self.features
            if feature.feature_type in [InputTypes.KNOWN, InputTypes.OBSERVED, InputTypes.STATIC]
            and feature.feature_embed_type == DataTypes.CONTINUOUS
        ]
        self.grouped = [group[1] for group in self.grouped]
        self.grouped = [
            group
            for group in self.grouped
            if len(group[group[self.time_feature] <= self.split]) >= self.encoder_length
            and len(group[group[self.time_feature] > self.split]) >= self.horizon
        ]

        self._cum_examples_in_group = np.cumsum(
            [(len(group[group[self.time_feature] > split]) - self.horizon) // self.stride + 1 for group in self.grouped]
        )

    def __len__(self):
        return self._cum_examples_in_group[-1]

    def __getitem__(self, idx):
        if idx > self._cum_examples_in_group[-1]:
            raise StopIteration
        g_idx = bisect(self._cum_examples_in_group, idx)
        e_idx = idx - self._cum_examples_in_group[g_idx - 1] if g_idx else idx
        group = self.grouped[g_idx]
        test = group[group[self.time_feature] > self.split]
        train = group[group[self.time_feature] <= self.split]
        test_slice = test[self.stride * e_idx : self.stride * e_idx + self.horizon]
        if (self.encoder_length - self.stride * e_idx) > 0:
            train_slice = train[-(self.encoder_length - self.stride * e_idx) :].append(
                test[max(0, self.stride * e_idx - self.encoder_length) : self.stride * e_idx]
            )
        else:
            train_slice = test[max(0, self.stride * e_idx - self.encoder_length) : self.stride * e_idx]

        train_out = {"endog": train_slice[self.endog], "exog": train_slice[self.exog]}

        test_out = {"endog": test_slice[self.endog], "exog": test_slice[self.exog], "id": test_slice[self.id_col_name]}
        if len(self.weight_features):
            test_out["weight"] = test_slice[self.weight_features]
        return train_out, test_out


def create_datasets(config):
    # XXX: We should probably fill all the fields in a config during it's construction with default
    # values so we avoid using `get`. This will reduce the number of bugs in the future.
    def select_dataset_class(config):
        binarized = config.dataset.get("binarized", False)
        graph_dataset = config.dataset.get("graph", False) and config.model.get("graph_eligible", False)

        if binarized and graph_dataset:
            specific_args = {
                "graph": os.path.join(config.dataset.dest_path, "graph.bin"),
                "graph_partitions": config.dataset.graph_partitions,
                "partition_joining_coef": config.dataset.partition_joining_coef,
            }
            return TemporalClusteredGraphDataset, specific_args
        elif binarized:
            return TSBinaryDataset, {}
        elif not binarized and graph_dataset:
            raise NotImplemented
        else:
            return TSDataset, {}

    common_args = {
        # XXX: calling this every time we need features in cumbersome. We could call this when the config
        # is constructed and enjoy not typig this line in every single function.
        "features": translate_features(config.dataset.features),
        "encoder_length": config.dataset.encoder_length,
        "example_length": config.dataset.example_length,
        "stride": config.dataset.get("stride", 1),
    }

    path_template = os.path.join(config.dataset.dest_path, "{{subset}}.{extension}")
    path_template = path_template.format(extension="bin" if config.dataset.get("binarized", False) else "csv")
    dataset_class, specific_args = select_dataset_class(config)

    train = dataset_class(path=path_template.format(subset="train"), **common_args, **specific_args)
    valid = dataset_class(path=path_template.format(subset="valid"), **common_args, **specific_args)
    test = dataset_class(path=path_template.format(subset="test"), **common_args, **specific_args)

    return train, valid, test


def map_dt(dt):
    if isinstance(dt, int):
        dt = dt
    elif isinstance(dt, ListConfig):
        dt = datetime.datetime(*dt)
    elif isinstance(dt, str):
        dt = datetime.datetime.strptime(dt, "%Y-%m-%d")
    return dt


class ClusteredGraphDataset(Dataset):
    def __init__(self, graph, graph_partitions=10, partition_joining_coef=2):
        if isinstance(graph, str):
            self.graph = pickle.load(open(graph, "rb"))
        else:
            self.graph = graph

        assert isinstance(graph_partitions, int) and graph_partitions > 0
        assert partition_joining_coef <= graph_partitions

        self.part_count = graph_partitions
        if graph_partitions > 1:
            self.partition = metis_partition_assignment(self.graph, self.part_count)
        else:
            self.partition = torch.zeros(self.graph.num_nodes(), dtype=torch.int64)
        self.joining_coef = partition_joining_coef

    def __len__(self):
        return math.comb(self.part_count, self.joining_coef)

    def __getitem__(self, idx):
        indicator = self.idx_to_combination(self.part_count, self.joining_coef, idx)
        c_ids = np.nonzero(indicator)[0]
        subgraph = self.get_subgraph(c_ids)
        return subgraph

    def get_subgraph(self, c_ids):
        ids = sum([self.partition == i for i in c_ids]).bool()
        return self.graph.subgraph(ids)

    def idx_to_combination(self, n, r, m):
        """
        n: int total number of elements
        r: int number of elements in combination
        m: int 0-based index of combination in reverse-lexicographic order
        
        Returns list - indicator vector of chosen elements
        """
        assert m < math.comb(n, r), "Index out of range"

        out = [0] * n
        while n > 0:
            if n > r and r >= 0:
                y = math.comb(n - 1, r)
            else:
                y = 0
            if m >= y:
                m -= y
                out[n - 1] = 1
                r -= 1
            n -= 1
        return out


class TemporalClusteredGraphDataset(ClusteredGraphDataset):
    def __init__(self, features, graph, path=None, encoder_length=52, example_length=54, stride=1, **kwargs):
        super().__init__(graph, **kwargs)
        assert example_length > encoder_length
        self.features = [i for i in features if i.feature_type != InputTypes.TIME]
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.stride = stride
        self.path = path

        self.feature_type_col_map = [
            np.array([i for i, f in enumerate(self.features) if (f.feature_type, f.feature_embed_type) == x])
            for x in FEAT_ORDER
        ]

        grouped = pickle.load(open(self.path, "rb"))
        # We assume that all the time series are of the same length and have the same set of features
        assert all([x.shape == grouped[0].shape for x in grouped])

        ndata = np.stack(grouped)
        self.ndata = {
            name: ndata[:, :, ids].view(dtype=np.float32).astype(DTYPE_MAP[f[1]])
            if not ids.size == 0
            else np.empty((*ndata.shape[:-1], 0))
            for name, f, ids in zip(FEAT_NAMES, FEAT_ORDER, self.feature_type_col_map)
        }

        self.t_dim = ndata.shape[1]
        self.n_timeslices = (self.t_dim - self.example_length + 1) // self.stride

    def __len__(self):
        # the number of possible subgraphs times the number of possible time slices
        return super().__len__() * self.n_timeslices

    def __getitem__(self, idx):
        g_idx = idx // self.n_timeslices
        t_idx = idx - g_idx * self.n_timeslices
        subgraph = super().__getitem__(g_idx)
        node_ids = np.array(subgraph.ndata["_ID"])
        for k, v in self.ndata.items():
            subgraph.ndata[k] = torch.from_numpy(
                v[node_ids, t_idx * self.stride : t_idx * self.stride + self.example_length, :]
            )

        return subgraph


def get_dataset_splits(df, config):
    if hasattr(config, "valid_boundary") and config.valid_boundary != None:
        forecast_len = config.example_length - config.encoder_length
        # The valid split is shifted from the train split by number of the forecast steps to the future.
        # The test split is shifted by the number of the forecast steps from the valid split
        train = []
        valid = []
        test = []
        valid_boundary = map_dt(config.valid_boundary)
        for _, group in df.groupby("_id_"):
            index = group[config.time_ids]
            _train = group.loc[index < valid_boundary]
            _valid = group.iloc[(len(_train) - config.encoder_length) : (len(_train) + forecast_len)]
            _test = group.iloc[(len(_train) - config.encoder_length + forecast_len) : (len(_train) + 2 * forecast_len)]
            train.append(_train)
            valid.append(_valid)
            test.append(_test)

        train = pd.concat(train, axis=0)
        valid = pd.concat(valid, axis=0)
        test = pd.concat(test, axis=0)

    elif df.dtypes[config.time_ids] not in [np.float64, np.int]:
        index = df[config.time_ids]

        train = df.loc[(index >= map_dt(config.train_range[0])) & (index < map_dt(config.train_range[1]))]
        valid = df.loc[(index >= map_dt(config.valid_range[0])) & (index < map_dt(config.valid_range[1]))]
        test = df.loc[(index >= map_dt(config.test_range[0])) & (index < map_dt(config.test_range[1]))]
    else:
        index = df[config.time_ids]
        train = df.loc[(index >= config.train_range[0]) & (index < config.train_range[1])]
        valid = df.loc[(index >= config.valid_range[0]) & (index < config.valid_range[1])]
        test = df.loc[(index >= config.test_range[0]) & (index < config.test_range[1])]

    return train, valid, test


def recombine_datasets(train, valid, test, config):
    if hasattr(config, "valid_boundary") and config.valid_boundary != None:
        forecast_len = config.example_length - config.encoder_length
        # The valid split is shifted from the train split by number of the forecast steps to the future.
        # The test split is shifted by the number of the forecast steps from the valid split
        train_temp = []
        valid_temp = []
        for g0, g1 in zip(train.groupby("_id_"), valid.groupby("_id_")):
            _train = g0[1].iloc[: -config.encoder_length]
            _valid = g1[1].iloc[:forecast_len]
            train_temp.append(_train)
            valid_temp.append(_valid)
        train = pd.concat(train_temp, axis=0)
        valid = pd.concat(valid_temp, axis=0)
    elif train.dtypes[config.time_ids] not in [np.float64, np.int]:

        train = train[train[config.time_ids] < map_dt(config.valid_range[0])]
        valid = valid[valid[config.time_ids] < map_dt(config.test_range[0])]
    else:
        train = train[train[config.time_ids] < config.valid_range[0]]
        valid = valid[valid[config.time_ids] < config.test_range[0]]
    return pd.concat((train, valid, test))


def flatten_ids(df, id_features):
    current_id = df[id_features[0]].astype("category").cat.codes + 1
    for additional_id in id_features[1:]:
        current_id = df[additional_id].astype("category").cat.codes * (current_id.max() + 1) + current_id + 1
    df["_id_"] = current_id.astype("category").cat.codes


def impute(df, config):
    # XXX This ensures that out scaling will have the same mean. We still need to check the variance
    # XXX does it work in place?
    if not (config.get("missing_data_label", False)):
        return df, None
    else:
        imp = SimpleImputer(missing_values=config.missing_data_label, strategy="mean")
        mask = df.applymap(lambda x: True if x == config.missing_data_label else False)
        data = df.values  # XXX this probably works in place. Check that!
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

    def inverse_transform_targets(self, values, ids):
        # Assuming single targets for now
        if len(self.target_scalers) > 0:

            scalers = list(self.target_scalers.values())[0]
            if self.scale_per_id:
                flat_values = values.flatten()
                flat_ids = np.repeat(ids, values.shape[1])
                df = pd.DataFrame({"id": flat_ids, "value": flat_values})
                df_list = []
                for identifier, sliced in df.groupby("id"):
                    df_list.append(scalers[identifier].inverse_transform(sliced["value"]))
                return np.concatenate(df_list, axis=None)
            else:
                flat_values = values.flatten()
                flat_values = scalers[""].inverse_transform(flat_values)
                return flat_values
        return values


def get_feature_splits(features):
    splits = {}
    splits["dates"] = [feature for feature in features if feature.feature_embed_type == DataTypes.DATE]
    splits["target_features"] = [feature for feature in features if feature.feature_type == InputTypes.TARGET]
    splits["time_feature"] = [feature for feature in features if feature.feature_type == InputTypes.TIME][0]
    splits["id_features"] = [feature for feature in features if feature.feature_type == InputTypes.ID]
    splits["input_categoricals"] = [
        feature
        for feature in features
        if feature.feature_embed_type == DataTypes.CATEGORICAL
        and feature.feature_type in [InputTypes.STATIC, InputTypes.KNOWN, InputTypes.OBSERVED]
    ]
    splits["input_continuous"] = [
        feature
        for feature in features
        if feature.feature_embed_type == DataTypes.CONTINUOUS
        and feature.feature_type in [InputTypes.STATIC, InputTypes.KNOWN, InputTypes.OBSERVED]
    ]
    return splits


def preprocess(config):
    config = config.dataset
    dest_path = config.dest_path
    features = translate_features(config["features"], preproc=True)
    feat_splits = get_feature_splits(features)

    print("Reading in data")
    df = pd.read_csv(config.source_path, parse_dates=[d.name for d in feat_splits["dates"]])
    print("Sorting on time feature")
    df = df.sort_values([feat_splits["time_feature"].name])
    f_names = [feature.name for feature in features] + [config.time_ids]
    df = df[list(set(f_names))]
    flatten_ids(df, [feature.name for feature in feat_splits["id_features"]])

    if config.get("missing_data_label", False):
        df = df.replace(config.get("missing_data_label"), np.NaN)
    df = df.dropna(subset=[t.name for t in feat_splits["target_features"]])
    print("Mapping categoricals to bounded range")

    for categorical in feat_splits["input_categoricals"]:
        df[categorical.name] = df[categorical.name].astype("category").cat.codes

    print("Splitting datasets")
    train, valid, test = get_dataset_splits(df, config)
    train = train.groupby("_id_").filter(lambda x: len(x) >= config.example_length)
    valid = valid.groupby("_id_").filter(lambda x: len(x) >= config.example_length)
    test = test.groupby("_id_").filter(lambda x: len(x) >= config.example_length)
    if hasattr(config, "valid_boundary") and config.valid_boundary != None:
        arriter = ["_id_"]
    else:
        arriter = [cat.name for cat in feat_splits["input_categoricals"]] + ["_id_"]

    if config.get("drop_unseen", False):
        for categorical in arriter:
            seen_values = train[categorical].unique()
            valid = valid[valid[categorical].isin(seen_values)]
            test = test[test[categorical].isin(seen_values)]
    print("Applying normalization")
    scaler = CompositeScaler(
        feat_splits["target_features"], feat_splits["input_continuous"], scale_per_id=config.scale_per_id
    )
    scaler.fit(train)

    train = scaler.transform(train)
    valid = scaler.transform(valid)
    test = scaler.transform(test)

    cont_features_names = [continuous.name for continuous in feat_splits["input_continuous"]]
    train[cont_features_names] = train[cont_features_names].replace(np.NaN, 10 ** 9)
    valid[cont_features_names] = valid[cont_features_names].replace(np.NaN, 10 ** 9)
    test[cont_features_names] = test[cont_features_names].replace(np.NaN, 10 ** 9)

    print("Saving processed data")
    os.makedirs(dest_path, exist_ok=True)

    train.to_csv(os.path.join(dest_path, "train.csv"))
    valid.to_csv(os.path.join(dest_path, "valid.csv"))
    test.to_csv(os.path.join(dest_path, "test.csv"))
    recombine_datasets(train, valid, test, config).to_csv(os.path.join(dest_path, "full.csv"))

    # Save relevant columns in binary form for faster dataloading
    # IMORTANT: We always expect id to be a single column indicating the complete timeseries
    # We also expect a copy of id in form of static categorical input!!!]]
    if config.get("binarized", False):
        col_names = ["_id_"] + [
            x.name
            for x in features
            if x.feature_embed_type != DataTypes.STR
            and x.feature_type != InputTypes.TIME
            and x.feature_type != InputTypes.ID
        ]
        grouped_train = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in train.groupby("_id_")]
        grouped_valid = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in valid.groupby("_id_")]
        grouped_test = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in test.groupby("_id_")]

        pickle.dump(grouped_train, open(os.path.join(dest_path, "train.bin"), "wb"))
        pickle.dump(grouped_valid, open(os.path.join(dest_path, "valid.bin"), "wb"))
        pickle.dump(grouped_test, open(os.path.join(dest_path, "test.bin"), "wb"))

    with open(os.path.join(dest_path, "composite_scaler.bin"), "wb") as f:
        pickle.dump(scaler, f)


def sample_data(dataset, num_samples):
    if num_samples < 0:
        return dataset
    else:
        return torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), size=num_samples, replace=False))
