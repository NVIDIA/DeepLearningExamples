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

import math
import os
import pickle
from bisect import bisect

import dgl
import numpy as np
import pandas as pd
import torch
from data.data_utils import InputTypes, DataTypes, FEAT_NAMES, FEAT_ORDER, DTYPE_MAP, translate_features
import dgl

from dgl.transform import metis_partition_assignment
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from data.xgb_util import load_xgb_df, feat_adder, data_label_split, select_test_group, target_shift, \
    xgb_multiID_preprocess
from bisect import bisect
from data.data_utils import InputTypes, DataTypes, FEAT_NAMES, FEAT_ORDER, DTYPE_MAP, translate_features, group_ids

class TSBaseDataset(Dataset):
    def __init__(self, features, df, encoder_length, example_length, stride=1, **kwargs):
        super().__init__()
        assert example_length > encoder_length
        self.features = features
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.stride = stride
        self.df = df
        self.load()
        self.features = [i for i in self.features if i.feature_type != InputTypes.TIME]
        self.feature_type_col_map = [
            [i for i, f in enumerate(self.features) if (f.feature_type, f.feature_embed_type) == x] for x in FEAT_ORDER
        ]

    def load(self):
        raise NotImplementedError

class TSDataset(TSBaseDataset):
    def __init__(self, features, df=None, encoder_length=52, example_length=54, stride=1, **kwargs):
        super().__init__(features, df, encoder_length, example_length, stride)
        self.grouped = [x for x in self.grouped if x.shape[0] >= self.example_length]
        self.group_lens = [(g.shape[0] - self.example_length + 1) // self.stride for g in self.grouped]
        self._cum_examples_in_group = np.cumsum(self.group_lens)

        self.grouped = [
            [
                arr[:, idxs].view(dtype=np.float32).astype(DTYPE_MAP[t[1]])
                for t, idxs in zip(FEAT_ORDER, self.feature_type_col_map)
            ]
            for arr in self.grouped
        ]

    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0)
        self.grouped = group_ids(data, self.features)

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
            torch.from_numpy(feat[e_idx * self.stride: e_idx * self.stride + self.example_length])
            if feat.size
            else torch.empty(0)
            for feat in group
        ]

        out = dict(zip(FEAT_NAMES, tensors))
        out["id"] = out["id"][0, :]
        return out

class TSBinaryDataset(TSDataset):
    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
            self.grouped = group_ids(data, self.features)
        else:
            self.grouped = pickle.load(open(self.df, "rb"))

class TSMultiIDDatasetBase(TSBaseDataset):
    def __init__(self,
            features, 
            df=None, 
            encoder_length=52, 
            example_length=54, 
            stride=1, 
            collumns_to_collapse=None,
            **kwargs
            ):
        super().__init__(features, df, encoder_length, example_length, stride)

        # This part is tricky: we want to do this only for training dataset and then apply the same changes to valid and test splits to maintain coherence.
        # We can't do this in the preprocessing step because many different dataset classes rely on the same csv file. Thus the first time dataset is created
        # if we pass empty list of collumns to collapse and populate it here. This list is a part for common argument set for the train, valid and test splits
        # so is maintained throughout construction of all the splits.
        if collumns_to_collapse is not None:
            if not collumns_to_collapse:
                for name, df in self.tables.items():
                    if df.eq(df.iloc[:, 0], axis=0).all().all():
                        self.tables[name] = df.iloc[:, :1]
                        collumns_to_collapse.append(name)
                    # Append dummy value to indicate that this this operation has already been performed
                    # This alleviates an edge case in which in train split we don't collapse any collumns and then we pass an empty list allowing collapse of
                    # collumns in valid and test sets.
                    collumns_to_collapse.append(None)
            else:
                for name in collumns_to_collapse:
                    if name is not None:
                        self.tables[name] = self.tables[name].iloc[:, :1]

        self.data = {}
        for fname, ftype in zip(FEAT_NAMES, FEAT_ORDER):
            names = [f.name for f in self.features if (f.feature_type, f.feature_embed_type) == ftype]
            if names:
                df = pd.concat([v for k,v in self.tables.items() if k in names], axis=1)
                self.data[fname] = df.values.astype(dtype=DTYPE_MAP[ftype[1]])
            else:
                self.data[fname] = None

        del self.tables
        self._n_timeslices = (next(len(df) for df in self.data.values() if df is not None) - self.example_length + 1) // self.stride

    def load(self):
        time_col_name = next(x.name for x in self.features if x.feature_type == InputTypes.TIME)
        id_col_name = next(x.name for x in self.features if x.feature_type == InputTypes.ID)
        if isinstance(self.df, pd.DataFrame):
            data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0)
        self.tables = {}
        for f in self.features:
            self.tables[f.name] = data.pivot(index=time_col_name, columns=id_col_name, values=f.name)

class TSMultiTargetDataset(TSMultiIDDatasetBase):

    def __len__(self):
        return self._n_timeslices 

    def __getitem__(self, idx):
        if idx < 0:
            idx = idx + len(self)
        if idx >= len(self) or idx < 0:
            raise IndexError

        out = {
              k: torch.from_numpy(v[idx * self.stride : idx * self.stride + self.example_length])
              if v is not None else torch.empty(0)
              for k,v in self.data.items()
              }

        return out

class TSMultiIDDataset(TSMultiIDDatasetBase):

    def __init__(self, features, df=None, encoder_length=52, example_length=54, stride=1, collumns_to_collapse=None, **kwargs):
        super().__init__(features, df, encoder_length, example_length, stride, collumns_to_collapse)

    def __len__(self):
        return self._n_timeslices * self.data['id'].shape[1]

    def __getitem__(self, idx):
        g_idx = idx // self._n_timeslices
        e_idx = idx - g_idx * self._n_timeslices

        targets = torch.from_numpy(self.data['target'][e_idx * self.stride : e_idx * self.stride + self.example_length])
        out = {
              k: torch.from_numpy(v[e_idx * self.stride : e_idx * self.stride + self.example_length, :])
              if v is not None else torch.empty(0)
              for k,v in self.data.items()
              }
        out['o_cont'] = torch.cat([out['o_cont'], targets], dim=-1)

        out['s_cat'] = out['s_cat'][:, g_idx].unsqueeze(1) if out['s_cat'].numel() else out['s_cat']
        out['s_cont'] = out['s_cont'][:, g_idx].unsqueeze(1) if out['s_cont'].numel() else out['s_cont']
        out['id'] = out['id'][:, g_idx]

        out['target'] = out['target'][:, g_idx].unsqueeze(1)
        out['weight'] = out['weight'][:, g_idx].unsqueeze(1) if out['weight'].numel() else out['weight']

        return out
        
class StatDataset(Dataset):
    def __init__(self, features, path_stat, df=None, encoder_length=52, example_length=54, stride=1, split=None, split_feature=None, ds_type=None):

        self.ds_type = ds_type
        if ds_type == "valid":
            return
        super().__init__()
        assert example_length > encoder_length, "Length of example longer than encoder length"
        assert split, "Split not given"
        assert ds_type in ["train", "test"]

        self.features = features
        self.time_feature = split_feature
        self.weight_features = [feature.name for feature in self.features if feature.feature_type == InputTypes.WEIGHT]
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.horizon = self.example_length - self.encoder_length
        self.stride = stride
        self.split = split
        self.id_col_name = next(x.name for x in self.features if x.feature_type == InputTypes.ID)
        self.col_dtypes = {v.name: DTYPE_MAP[v.feature_embed_type] for v in self.features}
        if isinstance(df, pd.DataFrame):
            self.data = df.astype(self.col_dtypes)
        else:
            self.data = pd.read_csv(os.path.join(path_stat, "full.csv"), dtype=self.col_dtypes)
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
        if self.ds_type == "valid":
            raise ValueError
        return self._cum_examples_in_group[-1]

    def __getitem__(self, idx):
        if self.ds_type == "valid":
            raise ValueError
        if idx > self._cum_examples_in_group[-1]:
            raise StopIteration
        g_idx = bisect(self._cum_examples_in_group, idx)
        e_idx = idx - self._cum_examples_in_group[g_idx - 1] if g_idx else idx
        group = self.grouped[g_idx]
        test = group[group[self.time_feature] > self.split]
        if self.ds_type == "test":
            test_slice = test[self.stride * e_idx: self.stride * e_idx + self.horizon]
            test_out = {"endog": test_slice[self.endog], "exog": test_slice[self.exog], "id": test_slice[self.id_col_name]}
            if len(self.weight_features):
                test_out["weight"] = test_slice[self.weight_features]
            return test_out
        else:
            train = group[group[self.time_feature] <= self.split]
            if (self.encoder_length - self.stride * e_idx) > 0:
                train_slice = train[-(self.encoder_length - self.stride * e_idx):].append(
                    test[max(0, self.stride * e_idx - self.encoder_length): self.stride * e_idx]
                )
            else:
                train_slice = test[max(0, self.stride * e_idx - self.encoder_length): self.stride * e_idx]

            train_out = {"endog": train_slice[self.endog], "exog": train_slice[self.exog]}
            return train_out


class XGBDataset(Dataset):
    def __init__(self, df, path_xgb, features_xgb, lag_features, moving_average_features, example_length, encoder_length, time_series_count, MultiID, ds_type, **kwargs):
        self.ds_type = ds_type
        features = features_xgb
        dest_path = df if isinstance(df, pd.DataFrame) else path_xgb
        self.encoder_length = encoder_length
        self.example_length = example_length
        lag_features_conf = lag_features
        self.lag_features = {}
        for feat in lag_features_conf:
            assert feat.get("min_value", None) is not None or feat.get("value", None) is not None
            if feat.get("min_value", None) is not None:
                assert feat.get("max_value", None) is not None and feat.get("min_value") > 0 and feat.get(
                    "max_value") > feat.get("min_value")
                self.lag_features[feat.name] = list(range(feat.get("min_value"), feat.get("max_value") + 1))
            else:
                self.lag_features[feat.name] = list(feat.value)
        moving_average_features_conf = moving_average_features
        self.moving_average_features = {}
        for feat in moving_average_features_conf:
            assert feat.get("window_size", None) is not None
            self.moving_average_features[feat.name] = self.moving_average_features.get(feat.name, []) + [
                feat.window_size]
        self.horizon = example_length - encoder_length
        self.target = [feature.name for feature in features if
                       feature.feature_type == "TARGET"]
        self.observed = [feature.name for feature in features if
                         feature.feature_type == "OBSERVED"]
        self.known = [feature.name for feature in features if
                      feature.feature_type in ["KNOWN", "STATIC"]]
        assert len(self.target) == 1, "Only 1 target feature is currently supported with xgboost"
        self.data = load_xgb_df(dest_path, features, ds_type)
        self.extra_columns = [[f'{k}_{i}' for i in v] for k, v in self.lag_features.items()]
        if MultiID:
            target = self.target[0]
            lag_target_value = self.lag_features.pop(target, [])
            for i in range(time_series_count):
                self.lag_features[f'{target}_{i}'] = lag_target_value
            self.moving_average_features[f'{target}_{i}'] = self.moving_average_features.pop(target, [])
            self.data = xgb_multiID_preprocess(self.data,  features, time_series_count) # XXX need to work with 
        self.data = feat_adder(self.data, self.lag_features, self.moving_average_features)

    def __getitem__(self, idx):
        if idx >= self.horizon:
            raise StopIteration
        data_step = self.data.copy()
        data_step = target_shift(data_step, self.target, self.known, idx)
        if self.ds_type == 'test':
            data_step = select_test_group(data_step, self.encoder_length, self.example_length)
        labels = data_label_split(data_step, [f'{i}_target' for i in self.target])
        return data_step, labels
    
    def __len__(self):
        return self.horizon

class ClusteredGraphDataset(Dataset):
    def __init__(self, graph, graph_partitions=10, partition_joining_coef=2, **kwargs):
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
    def __init__(self, features, graph, df=None, encoder_length=52, example_length=54, stride=1, **kwargs):
        super().__init__(graph, **kwargs)
        assert example_length > encoder_length
        self.features = [i for i in features if i.feature_type != InputTypes.TIME]
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.stride = stride
        self.df = df

        self.feature_type_col_map = [
            np.array([i for i, f in enumerate(self.features) if (f.feature_type, f.feature_embed_type) == x])
            for x in FEAT_ORDER
        ]
        if isinstance(df, pd.DataFrame):
            data = self.df
            grouped = group_ids(data, self.features)
        else:
            grouped = pickle.load(open(self.df, "rb"))
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
                v[node_ids, t_idx * self.stride: t_idx * self.stride + self.example_length, :]
            )

        return subgraph



def create_datasets(config, input_df=None):
    def select_dataset_class(config):
        binarized = config.get("binarized", False)
        graph_dataset = config.get("construct_graph", False)
        multi_id_dataset = config.get("MultiID", False)
        single_target = config.get('single_target', False)
        if config.get("xgb", False):
            specific_args = {
                "path_xgb": config.dest_path,
                "features_xgb": config.features,
                "lag_features": config.get("lag_features", []),
                "moving_average_features": config.get("moving_average_features", []),
                "time_series_count": config.time_series_count,
                "MultiID": config.get("MultiID", False)
            }
            return XGBDataset, specific_args

        if config.get("stat", False):

            specific_args = {
                "path_stat": config.dest_path,
                "split": config.test_range[0],
                "split_feature": config.time_ids
            }
            return StatDataset, specific_args
        if binarized and graph_dataset:
            specific_args = {
                "graph": os.path.join(config.dest_path, "graph.bin"),
                "graph_partitions": config.graph_partitions,
                "partition_joining_coef": config.partition_joining_coef,
            }
            return TemporalClusteredGraphDataset, specific_args
        elif binarized and multi_id_dataset:
            raise NotImplementedError
        elif binarized:
            return TSBinaryDataset, {}
        elif not binarized and graph_dataset:
            raise NotImplementedError
        elif not binarized and multi_id_dataset and not single_target:
            specific_args = {}
            if config.get('collapse_identical_columns', False):
                specific_args['collumns_to_collapse'] = []
            return TSMultiTargetDataset, specific_args
        elif not binarized and multi_id_dataset and single_target:
            specific_args = {}
            if config.get('collapse_identical_columns', False):
                specific_args['collumns_to_collapse'] = []
            return TSMultiIDDataset, specific_args
        else:
            return TSDataset, {}

    common_args = {
        "features": translate_features(config.features),
        "encoder_length": config.encoder_length,
        "example_length": config.example_length,
        "stride": config.get("stride", 1),
    }

    dataset_class, specific_args = select_dataset_class(config)

    if input_df is not None:
        print("Input DataFrame provided to create_datasets functions")
        print("Warning: Please make sure the dataframe is preprocessed")
        test = dataset_class(df=input_df, **common_args, **specific_args, ds_type='test')
        train = None
        valid = None
    else:
        path_template = os.path.join(config.dest_path, "{{subset}}.{extension}")
        path_template = path_template.format(extension="bin" if config.get("binarized", False) else "csv")

        train = dataset_class(df=path_template.format(subset="train"), **common_args, **specific_args, ds_type="train")
        valid = dataset_class(df=path_template.format(subset="valid"), **common_args, **specific_args, ds_type="valid")
        test = dataset_class(df=path_template.format(subset="test"), **common_args, **specific_args, ds_type="test")
        if not (config.get("xgb", False) or config.get("stat", False)):
            train = sample_data(train, config.get("train_samples", -1))
            valid = sample_data(valid, config.get("valid_samples", -1))

    return train, valid, test


def sample_data(dataset, num_samples):
    if num_samples < 0:
        return dataset
    else:
        return torch.utils.data.Subset(dataset,
                                       np.random.choice(np.arange(len(dataset)), size=num_samples, replace=False))


def get_collate_fn(model_type, encoder_length, test=False):
    allowed_types = ['default', 'graph', 'autoregressive']
    if model_type not in allowed_types:
        raise ValueError(f'Model type has to be one of {allowed_types}')

    def collate_graph(samples):
        """A collater used for GNNs"""
        batch = dgl.batch(samples)
        labels = batch.ndata["target"][:, encoder_length:, :]
        weights = batch.ndata['weight']
        if weights is not None and weights.numel():
            weights = weights[:, encoder_length :, :]
        return batch, labels, weights

    def collate_ar(samples):
        batch = default_collate(samples)
        labels = batch["target"]
        weights = batch['weight']
        return batch, labels, weights

    def collate_dict(samples):
        """Default TSPP collater"""
        batch = default_collate(samples)
        labels = batch["target"][:, encoder_length:, :]
        weights = batch['weight']
        if weights is not None and weights.numel():
            weights = weights[:, encoder_length:, :]
        return batch, labels, weights

    if model_type == 'graph':
        return collate_graph
    elif model_type == 'autoregressive' and not test:
        return collate_ar
    else:
        return collate_dict
