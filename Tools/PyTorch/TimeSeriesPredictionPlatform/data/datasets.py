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

from abc import abstractmethod
from typing import Union, List
import enum
import warnings
import mmap
import math
import os
import pickle
import logging
from bisect import bisect
from collections import Counter, namedtuple, Iterable
from itertools import product

import numpy as np
import pandas as pd
import torch
import dgl

from dgl import metis_partition_assignment
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from data.xgb_util import feat_adder, data_label_split, select_test_group, target_shift, xgb_multiID_preprocess
from bisect import bisect, bisect_left
from data.data_utils import InputTypes, DataTypes, FEAT_NAME_MAP, DTYPE_MAP, translate_features, group_ids, get_alignment_compliment_bytes


ArgDesc = namedtuple(
    'ArgDesc',
    ['name', 'required', 'default', 'extractor'],
)

DatasetDesc = namedtuple(
    'DatasetDesc',
    ['type',  # ether xgb, stat or default (DL) dataset
     'data_layout',  # e.g. binarized
     'entity_type',  # e.g. graph, multi_id, ect
     'target_type'],  # single or multi target
)

class DatasetType(enum.IntEnum):
    DL = 0
    XGB = 1
    STAT = 2

    @staticmethod
    def parse(config):
        dataset_type = DatasetType.DL
        if config.get('xgb', False):
            dataset_type = DatasetType.XGB
        elif config.get('stat', False):
            dataset_type = DatasetType.STAT
        return dataset_type

class DataLayout(enum.IntEnum):
    DEFAULT = 0
    BINARIZED = 1
    MEMORY_MAPPED = 2

    @staticmethod
    def parse(config):
        data_layout = DataLayout.DEFAULT
        if config.get('memory_mapped', False):
            data_layout = DataLayout.MEMORY_MAPPED
        elif config.get('binarized', False):
            data_layout = DataLayout.BINARIZED
        return data_layout

class EntityType(enum.IntEnum):
    DEFAULT = 0
    GRAPH = 1
    SHARDED = 2
    MULTIID = 3

    @staticmethod
    def parse(config):
        entity_type = EntityType.DEFAULT
        if config.get("construct_graph", False):
            entity_type = EntityType.GRAPH
        elif config.get('sharded', False):
            entity_type = EntityType.SHARDED
        elif config.get("MultiID", False):
            entity_type = EntityType.MULTIID
        return entity_type

class TargetType(enum.IntEnum):
    SINGLE = 0
    MULTI = 1

    @staticmethod
    def parse(config):
        target_type = TargetType.MULTI if config.get("MultiID", False) else TargetType.SINGLE
        if config.get('single_target', False) or config.get('construct_graph', False):
            target_type = TargetType.SINGLE
        return target_type


class BaseDataset(Dataset):
    configuration_args = [
        ArgDesc(name='features', required=True, default=None, extractor=lambda config: translate_features(config.features)),
        ArgDesc(name='encoder_length', required=True, default=None, extractor=None),
        ArgDesc(name='example_length', required=True, default=None, extractor=None),
        ArgDesc(name='stride', required=False, default=1, extractor=None)
    ]

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError


class DatasetFactory:
    _dataset_registry = {}

    @classmethod
    def register(cls, type: Union[DatasetType, List[DatasetType]], 
                 data_layout: Union[DataLayout, List[DataLayout]], 
                 entity_type: Union[EntityType, List[EntityType]], 
                 target_type: Union[TargetType, List[TargetType]]
                ):
        def inner_wrapper(wrapped_class: BaseDataset):
            descriptions = [d if isinstance(d, Iterable) else [d] 
                     for d in (type, data_layout, entity_type, target_type)]
            for desc in product(*descriptions):
                if desc in cls._dataset_registry:
                    raise ValueError(f'{wrapped_class.__class__.__name__} and {cls._dataset_registry[desc].__class__.__name__} '
                                     'datasets match same description. Please, resolve the conflict manually.')
                cls._dataset_registry[desc] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def construct_dataset(cls, dataset_desc, df, config):
        if dataset_desc not in cls._dataset_registry:
            raise ValueError(f'Failed to create dataset: There is no dataset that matches description {dataset_desc}.')
        dataset_class: BaseDataset = cls._dataset_registry[dataset_desc]
        dataset_kwargs = {}
        for arg in dataset_class.configuration_args:
            val = arg.default
            if arg.extractor:
                try:
                    val = arg.extractor(config)
                except Exception as e:
                    if arg.required:
                        raise
                    else:
                        print('Encountered error during config parsing', e)
            else:
                if arg.required:
                    val = config[arg.name]
                else:
                    val = config.get(arg.name, arg.default) 
            dataset_kwargs[arg.name] = val
        
        ds = dataset_class(df=df, **dataset_kwargs)
        return ds
    

class TSBaseDataset(BaseDataset):
    def __init__(self, features, df, encoder_length, example_length, stride=1, **kwargs):
        super().__init__()
        assert example_length > encoder_length
        self.features = features
        self.time_feat = [i for i in self.features if i.feature_type == InputTypes.TIME][0]
        self.encoder_length = encoder_length
        self.example_length = example_length
        self.stride = stride
        self.df = df
        self.load()

    @abstractmethod
    def load(self):
        raise NotImplementedError


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.DEFAULT,
    entity_type=EntityType.DEFAULT,
    target_type=TargetType.SINGLE
)
class TSDataset(TSBaseDataset):
    def __init__(self, features, df=None, encoder_length=52, example_length=54, stride=1, **kwargs):
        super().__init__(features, df, encoder_length, example_length, stride)

        group_lens = (self.group_sizes - self.example_length + 1) // self.stride
        self._cum_examples_in_group = np.cumsum(group_lens)
        self._group_last_idx = np.cumsum(self.group_sizes)

    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0, engine='pyarrow')
        self.grouped, self.group_sizes = group_ids(data, self.features)

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
        offset = self._group_last_idx[g_idx - 1] if g_idx else 0
        e_idx = idx - self._cum_examples_in_group[g_idx - 1] if g_idx else idx

        start = offset + e_idx * self.stride
        end = offset + e_idx * self.stride + self.example_length
        assert end <= self._group_last_idx[g_idx]

        out = {
            name: torch.from_numpy(feat[start:end])
            for name, feat in zip(FEAT_NAME_MAP.keys(), self.grouped)
        }

        out["id"] = out["id"][0, :]
        out["timestamp"] = out["timestamp"][0, :]

        return out


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.DEFAULT,
    entity_type=EntityType.SHARDED,
    target_type=TargetType.SINGLE
)
class TSShardedDataset(TSBaseDataset):
    """
    Experimental class.
    """
    def __init__(self, features, df=None, encoder_length=52, example_length=54, stride=1, **kwargs):
        super().__init__(features, df, encoder_length, example_length, stride)

    def autodetect_shards(self, df):
        time_feat = [i for i in self.features if i.feature_type == InputTypes.TIME][0]
        time_diffs = df[time_feat.name].diff()
        counter = Counter(time_diffs)
        timestep = counter.most_common()[0][0]

        # create groups based on consecutive time differences
        groups = (time_diffs != timestep).cumsum()
        # group the DataFrame by the groups and create a dictionary of continuous blocks
        shards = {group: data for group, data in df.groupby(groups) if len(data) >= self.encoder_length}
        return shards

    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0, engine='pyarrow')

        shards = self.autodetect_shards(data)

        self.shards = [TSDataset(self.features, 
                                 df=shard, 
                                 encoder_length=self.encoder_length, 
                                 example_length=self.example_length,
                                 stride=1)
                                 for shard in shards.values()
        ]
        self._cum_shards_len = np.cumsum([len(ds) for ds in self.shards])

    def __len__(self):
        return self._cum_shards_len[-1]

    def __getitem__(self, idx):
        g_idx = bisect(self._cum_shards_len, idx)
        e_idx = idx - self._cum_shards_len[g_idx - 1] if g_idx else idx

        return self.shards[g_idx][e_idx]


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.BINARIZED,
    entity_type=EntityType.DEFAULT,
    target_type=TargetType.SINGLE
)
class TSBinaryDataset(TSDataset):
    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
            self.grouped, self.group_sizes = group_ids(data, self.features)
        else:
            with open(self.df, "rb") as f:
                metadata = pickle.load(f)
                self.group_sizes = metadata['group_sizes']
                self.grouped = []
                for dtype, shape, _ in metadata['col_desc']:
                    offset = get_alignment_compliment_bytes(f.tell(), dtype)
                    self.grouped.append(
                        np.fromfile(f, dtype=dtype, count=np.prod(shape), offset=offset).reshape(*shape)
                    )


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.MEMORY_MAPPED,
    entity_type=EntityType.DEFAULT,
    target_type=TargetType.SINGLE
)
class TSMemoryMappedDataset(TSDataset):
    warnings.filterwarnings('ignore', category=UserWarning, message='The given NumPy array is not writable,')

    def load(self):
        if isinstance(self.df, pd.DataFrame):
            raise ValueError(f'{self.__class__.__name__} does not support loading from DataFrame')
        
        f = open(self.df, "rb")
        metadata = pickle.load(f)
        self.group_sizes = metadata['group_sizes']

        offset = f.tell()
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # try to enable huge pages to improve performance
            buf.madvise(mmap.MADV_HUGEPAGE)
        except Exception:
            logging.info("Failed to enable huge pages on mapped dataset")

        # it would be nice for the OS to load some pages ahead
        # in case they are on an actual file system and not shmem/tmpfs
        buf.madvise(mmap.MADV_WILLNEED)

        self.grouped = []
        for dtype, shape, nbytes in metadata['col_desc']:
            offset += get_alignment_compliment_bytes(offset, dtype)
            self.grouped.append(
                np.frombuffer(buffer=buf, dtype=dtype, count=np.prod(shape), offset=offset).reshape(*shape)
            )
            offset += nbytes


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
        for fname, ftype in FEAT_NAME_MAP.items():
            names = [f.name for f in self.features if (f.feature_type, f.feature_embed_type) == ftype]
            if names:
                self.data[fname] = [v.values.astype(dtype=DTYPE_MAP[ftype[1]]) for k,v in self.tables.items() if k in names]
            else:
                self.data[fname] = None

        del self.tables
        self._n_timeslices = (next(len(x[0]) for x in self.data.values() if x is not None) - self.example_length + 1) // self.stride

    def load(self):
        time_col_name = next(x.name for x in self.features if x.feature_type == InputTypes.TIME)
        id_col_name = next(x.name for x in self.features if x.feature_type == InputTypes.ID)
        if isinstance(self.df, pd.DataFrame):
            data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0, engine='pyarrow')
        self.tables = {}
        for f in self.features:
            self.tables[f.name] = data.pivot(index=time_col_name, columns=id_col_name, values=f.name)


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.DEFAULT,
    entity_type=EntityType.MULTIID,
    target_type=TargetType.MULTI
)
class TSMultiTargetDataset(TSMultiIDDatasetBase):
    configuration_args = (
        TSMultiIDDatasetBase.configuration_args + 
        [
            ArgDesc(name='collumns_to_collapse', required=False, default=None, extractor=lambda config: [] if config.get('collapse_identical_columns', False) else None)
        ]
    )
    def __init__(self, *args, **kwargs):
        assert kwargs.get('columns_to_collapse') is None, "Can't use TSMultiTargetDataset with collapse_identical_columns=True"
        super().__init__(*args, **kwargs)
        self.data = {k: np.stack(v, axis=-1) if v is not None else None for k, v in self.data.items()}


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

        # There is only one id column, so squeeze dimension which was produced by torch.stack
        out['id'] = out['id'].squeeze(-1)
        out['timestamp'] = out['timestamp'].squeeze(-1)

        return out


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.DEFAULT,
    entity_type=EntityType.MULTIID,
    target_type=TargetType.SINGLE
)
class TSMultiIDDataset(TSMultiIDDatasetBase):
    configuration_args = (
        TSMultiIDDatasetBase.configuration_args + 
        [
            ArgDesc(name='collumns_to_collapse', required=False, default=None, extractor=lambda config: [] if config.get('collapse_identical_columns', False) else None)
        ]
    )
    def __init__(self, features, df=None, encoder_length=52, example_length=54, stride=1, collumns_to_collapse=None, **kwargs):
        super().__init__(features, df, encoder_length, example_length, stride, collumns_to_collapse)
        self.data = {k: np.concatenate(v, axis=-1) if v is not None else None for k, v in self.data.items()}

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


@DatasetFactory.register(
    type=DatasetType.STAT,
    data_layout=DataLayout.DEFAULT,
    entity_type=EntityType.DEFAULT,
    target_type=TargetType.SINGLE
) 
class StatDataset(TSBaseDataset):
    configuration_args = (
        TSBaseDataset.configuration_args + 
        [
            ArgDesc(name='use_last', required=False, default=0, extractor=None)
        ]
    )
    def __init__(self, features, df=None, encoder_length=52, example_length=54, stride=1, use_last=0):
        self.use_last = use_last
        super().__init__(features, df, encoder_length, example_length, stride)
        self.test = False

        self.horizon = self.example_length - self.encoder_length
        feat_names = list(FEAT_NAME_MAP.keys())
        self.id_col_id = feat_names.index('id')
        self.weight_col_id = feat_names.index('weight')
        self.endog_col_id = feat_names.index('target')
        self.exog_col_id = [i for i, feat in enumerate(feat_names) if feat.endswith('cont')]

        self._group_last_idx = np.cumsum(self.group_sizes)
        self._cum_examples_in_group = np.cumsum((self.group_sizes - self.horizon + 1) // self.stride)
    
    def load(self):
        if isinstance(self.df, pd.DataFrame):
            self.data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0, engine='pyarrow')
        self.grouped, self.group_sizes = group_ids(data, self.features)

    def __len__(self):
        return self._cum_examples_in_group[-1] if self.test else len(self.group_sizes)

    def __getitem__(self, idx):
        if ((self.test and idx > self._cum_examples_in_group[-1]) or 
            (not self.test and idx > len(self.group_sizes))):
            raise StopIteration
        
        if not self.test:
            start = self._group_last_idx[idx - 1] if idx else 0
            end = self._group_last_idx[idx]
            if self.use_last > 0:
                start = end - self.use_last
            
            update_start = 0
            update_end = 0
        else:
            g_idx = bisect(self._cum_examples_in_group, idx)
            offset = self._group_last_idx[g_idx - 1] if g_idx else 0
            e_idx = idx - self._cum_examples_in_group[g_idx - 1] if g_idx else idx

            start = offset + e_idx * self.stride
            end = offset + e_idx * self.stride + self.horizon
            assert end <= self._group_last_idx[g_idx]
            update_start = (e_idx - 1 if e_idx else 0) * self.stride
            update_end = e_idx * self.stride
        
        out = {
            'endog': self.grouped[self.endog_col_id][start:end],
            'exog': np.hstack(self.grouped[i][start:end] for i in self.exog_col_id),
            'id': self.grouped[self.id_col_id][start].item(),
            'weight': self.grouped[self.weight_col_id][start:end],
            'endog_update': self.grouped[self.endog_col_id][update_start:update_end],
            'exog_update': np.hstack(self.grouped[i][update_start:update_end] for i in self.exog_col_id),
        }
        return out

@DatasetFactory.register(
    type=DatasetType.STAT,
    data_layout=DataLayout.BINARIZED,
    entity_type=EntityType.DEFAULT,
    target_type=TargetType.SINGLE
) 
class BinaryStatDataset(StatDataset):
    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
            self.grouped, self.group_sizes = group_ids(data, self.features)
        else:
            with open(self.df, "rb") as f:
                metadata = pickle.load(f)
                self.group_sizes = metadata['group_sizes']
                self.grouped = []
                for dtype, shape, _ in metadata['col_desc']:
                    offset = get_alignment_compliment_bytes(f.tell(), dtype)
                    self.grouped.append(
                        np.fromfile(f, dtype=dtype, count=np.prod(shape), offset=offset).reshape(*shape)
                    )

@DatasetFactory.register(
    type=DatasetType.STAT,
    data_layout=DataLayout.MEMORY_MAPPED,
    entity_type=EntityType.DEFAULT,
    target_type=TargetType.SINGLE
) 
class TSMemoryMappedDataset(StatDataset):
    def load(self):
        if isinstance(self.df, pd.DataFrame):
            raise ValueError(f'{self.__class__.__name__} does not support loading from DataFrame')
        
        f = open(self.df, "rb")
        metadata = pickle.load(f)
        self.group_sizes = metadata['group_sizes']

        offset = f.tell()
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # try to enable huge pages to improve performance
            buf.madvise(mmap.MADV_HUGEPAGE)
        except Exception:
            logging.info("Failed to enable huge pages on mapped dataset")

        # it would be nice for the OS to load some pages ahead
        # in case they are on an actual file system and not shmem/tmpfs
        buf.madvise(mmap.MADV_WILLNEED)

        self.grouped = []
        for dtype, shape, nbytes in metadata['col_desc']:
            offset += get_alignment_compliment_bytes(offset, dtype)
            self.grouped.append(
                np.frombuffer(buffer=buf, dtype=dtype, count=np.prod(shape), offset=offset).reshape(*shape)
            )
            offset += nbytes


@DatasetFactory.register(
    type=DatasetType.XGB,
    data_layout=DataLayout.DEFAULT,
    entity_type=[EntityType.DEFAULT, EntityType.MULTIID],
    target_type=TargetType.SINGLE
) 
class XGBDataset(TSBaseDataset):
    configuration_args = (
        TSBaseDataset.configuration_args + 
        [
            ArgDesc(name='lag_features', required=False, default=[], extractor=None),
            ArgDesc(name='moving_average_features', required=False, default=[], extractor=None),
            ArgDesc(name='MultiID', required=False, default=False, extractor=None),
        ]
    )

    def __init__(self, features, df, encoder_length, example_length, lag_features, moving_average_features, MultiID, **kwargs):
        super().__init__(features, df, encoder_length, example_length, stride=1)

        self.test = False

        self.horizon = example_length - encoder_length
        self.target = [feature.name for feature in features if
                       feature.feature_type == InputTypes.TARGET]
        self.time_feat = [feature.name for feature in features if
                       feature.feature_type == InputTypes.TIME]

        # Filter out special features
        self.features = [f for f in self.features if not f.feature_type in (InputTypes.TIME, InputTypes.WEIGHT, InputTypes.ID)]

        self.observed = [feature.name for feature in features if
                         feature.feature_type == InputTypes.OBSERVED]
        self.known = [feature.name for feature in features if
                      feature.feature_type in [InputTypes.KNOWN, InputTypes.STATIC]]
        
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

        if MultiID:
            target = self.target[0]
            lag_target_value = self.lag_features.pop(target, [])

            time_series_count = self.data['_id_'].nunique()
            for i in range(time_series_count):
                self.lag_features[f'{target}_{i}'] = lag_target_value
            self.moving_average_features[f'{target}_{i}'] = self.moving_average_features.pop(target, [])
            self.data = xgb_multiID_preprocess(self.data,  self.time_feat[0], target)

        self.data = feat_adder(self.data, self.lag_features, self.moving_average_features)
        self.data = self.data.loc[:, sorted(self.data.columns)]

    def load(self):
        if isinstance(self.df, pd.DataFrame):
            data = self.df
        else:
            data = pd.read_csv(self.df, index_col=0, engine='pyarrow')

        all_features = {f.name for f in self.features}
        data = data[all_features]
        data = data.select_dtypes(exclude='object')
        self.data = data

    def __getitem__(self, idx):
        if idx >= self.horizon:
            raise StopIteration
        data_step = self.data.copy()
        data_step = target_shift(data_step, self.target, [], idx)
        if self.test:
            data_step = select_test_group(data_step, self.encoder_length, self.example_length)
        labels = data_label_split(data_step, [f'{i}_target' for i in self.target])
        return data_step, labels
    
    def __len__(self):
        return self.horizon


@DatasetFactory.register(
    type=DatasetType.DL,
    data_layout=DataLayout.DEFAULT,
    entity_type=EntityType.GRAPH,
    target_type=TargetType.SINGLE
) 
class TemporalClusteredGraphDataset(TSMultiIDDatasetBase):
    configuration_args = (
        TSMultiIDDatasetBase.configuration_args + 
        [
            ArgDesc(name='graph', required=True, default=None, extractor=lambda config: os.path.join(config.dest_path, config.graph)),
            ArgDesc(name='graph_partitions', required=True, default=None, extractor=None),
            ArgDesc(name='partition_joining_coef', required=True, default=None, extractor=None),
        ]
    )
    def __init__(self, features, graph, df=None, encoder_length=52, example_length=54, stride=1, graph_partitions=1, partition_joining_coef=1, **kwargs):
        assert isinstance(graph_partitions, int) and graph_partitions > 0
        assert partition_joining_coef <= graph_partitions
        assert graph is not None

        super().__init__(features, df, encoder_length, example_length, stride, collumns_to_collapse=None)

        if isinstance(graph, str):
            self.graph = pickle.load(open(graph, "rb"))
            if isinstance(self.graph, np.ndarray):
                edges = np.nonzero(self.graph)
                weights = self.graph[edges]
                self.graph = dgl.graph(edges)
                self.graph.edata['w'] = torch.tensor(weights)
        else:
            self.graph = graph

        self.part_count = graph_partitions
        if graph_partitions > 1:
            self.partition = metis_partition_assignment(self.graph, self.part_count)
        else:
            self.partition = torch.zeros(self.graph.num_nodes(), dtype=torch.int64)
        self.joining_coef = partition_joining_coef

        for k,v in self.data.items():
            if v is not None:
                self.data[k] = np.stack(self.data[k], axis=1).transpose(2,0,1)

    def __len__(self):
        return math.comb(self.part_count, self.joining_coef) * self._n_timeslices

    def __getitem__(self, idx):
        g_idx = idx // self._n_timeslices
        t_idx = idx - g_idx * self._n_timeslices
        subgraph = self.get_subgraph(g_idx)
        node_ids = np.array(subgraph.ndata["_ID"])
        for k, v in self.data.items():
            subgraph.ndata[k] = torch.from_numpy(
                v[node_ids, t_idx * self.stride: t_idx * self.stride + self.example_length, :]
            ) if v is not None else torch.empty((self.graph.num_nodes(),0))

        subgraph.ndata['id'] = subgraph.ndata['id'][:,0,:]

        return subgraph

    def get_subgraph(self, idx):
        indicator = self.idx_to_combination(self.part_count, self.joining_coef, idx)
        c_ids = np.nonzero(indicator)[0]
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


def _parse_dataset_description(config):
    dataset_type = DatasetType.parse(config=config)
    data_layout = DataLayout.parse(config=config)
    entity_type = EntityType.parse(config=config)
    target_type = TargetType.parse(config=config)
    
    return DatasetDesc(
        type=dataset_type,
        data_layout=data_layout,
        entity_type=entity_type,
        target_type=target_type
    )

def create_datasets(config, input_df=None):
    dataset_desc = _parse_dataset_description(config)
    if input_df is not None:
        print("Input DataFrame provided to create_datasets functions")
        print("Warning: Please make sure the dataframe is preprocessed")
        test = DatasetFactory.construct_dataset(dataset_desc, df=input_df, config=config)
        train = None
        valid = None
    else:
        path_template = os.path.join(config.dest_path, "{{subset}}.{extension}")
        path_template = path_template.format(extension="bin" if config.get("binarized", False) or config.get("memory_mapped", False) else "csv")

        train = DatasetFactory.construct_dataset(dataset_desc, 
                                                 df=path_template.format(subset="train" if dataset_desc.type is not DatasetType.STAT else "train_stat"), 
                                                 config=config, 
                                                 )
        
        valid = DatasetFactory.construct_dataset(dataset_desc, 
                                                 df=path_template.format(subset="valid"), 
                                                 config=config, 
                                                 ) if dataset_desc.type is not DatasetType.STAT else None
        
        test = DatasetFactory.construct_dataset(dataset_desc, 
                                                 df=path_template.format(subset="test" if dataset_desc.type is not DatasetType.STAT else "test_stat"), 
                                                 config=config, 
                                                 )
        
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

    def collate_dict(samples):
        """Default TSPP collater"""
        batch = default_collate(samples)
        labels = batch["target"][:, encoder_length :, :]
        if test:
            labels = labels.clone()
            batch['target'][:, encoder_length :, :] = 0
            if batch['o_cat'].numel():
                batch['o_cat'][:, encoder_length :, :] = 0
            if batch['o_cont'].numel():
                batch['o_cont'][:, encoder_length :, :] = 0
        weights = batch['weight']
        if weights is not None and weights.numel():
            weights = weights[:, encoder_length :, :]
        return batch, labels, weights

    def collate_ar(samples):
        """A collater for autoregressive models"""
        batch = default_collate(samples)
        labels = batch["target"]
        weights = batch['weight']
        if test:
            labels = labels.clone()
            labels = labels[:, encoder_length:, :]
            batch['target'][:, encoder_length:, :] = 0
            if batch['o_cat'].numel():
                batch['o_cat'][:, encoder_length:, :] = 0
            if batch['o_cont'].numel():
                batch['o_cont'][:, encoder_length:, :] = 0
            if weights is not None and weights.numel():
                weights = weights[:, encoder_length:, :]

        return batch, labels, weights

    if model_type == 'graph':
        return collate_graph
    if model_type == 'autoregressive':
        return collate_ar
    else:
        return collate_dict
