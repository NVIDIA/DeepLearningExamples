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
################################
# Copyright 2021 The Google Research Authors.
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

import os
import math
import pickle
import enum
import datetime

from collections import namedtuple, OrderedDict

import sklearn.preprocessing
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from bisect import bisect

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, RandomSampler
from torch.utils.data.dataloader import default_collate

class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3

class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED = 1
    KNOWN = 2
    STATIC = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index

FeatureSpec = namedtuple('FeatureSpec', ['name', 'feature_type', 'feature_embed_type'])
DTYPE_MAP = {
        DataTypes.CONTINUOUS : np.float32,
        DataTypes.CATEGORICAL : np.int64,
        DataTypes.DATE:'datetime64[ns]',
        DataTypes.STR: str
        }

FEAT_ORDER = [
        (InputTypes.STATIC, DataTypes.CATEGORICAL),
        (InputTypes.STATIC, DataTypes.CONTINUOUS),
        (InputTypes.KNOWN, DataTypes.CATEGORICAL),
        (InputTypes.KNOWN, DataTypes.CONTINUOUS),
        (InputTypes.OBSERVED, DataTypes.CATEGORICAL),
        (InputTypes.OBSERVED, DataTypes.CONTINUOUS),
        (InputTypes.TARGET, DataTypes.CONTINUOUS),
        (InputTypes.ID, DataTypes.CATEGORICAL)
        ]

FEAT_NAMES = ['s_cat' , 's_cont' , 'k_cat' , 'k_cont' , 'o_cat' , 'o_cont' , 'target', 'id']
DEFAULT_ID_COL = 'id'

class TFTBinaryDataset(Dataset):
    def __init__(self, path, config):
        super(TFTBinaryDataset).__init__()
        self.features = [x for x in config.features if x.feature_embed_type != DataTypes.DATE]
        self.example_length = config.example_length
        self.stride = config.dataset_stride

        self.grouped = pickle.load(open(path, 'rb'))
        self.grouped = [x for x in self.grouped if x.shape[0] >= self.example_length]
        self._cum_examples_in_group = np.cumsum([(g.shape[0] - self.example_length + 1)//self.stride for g in self.grouped])


        self.feature_type_col_map = [[i for i,f in enumerate(self.features) if (f.feature_type, f.feature_embed_type) == x] for x in FEAT_ORDER]

        # The list comprehension below is an elaborate way of rearranging data into correct order,
        # simultaneously doing casting to proper types. Probably can be written neater
        self.grouped = [
                [
                    arr[:, idxs].view(dtype=np.float32).astype(DTYPE_MAP[t[1]]) 
                    for t, idxs in zip(FEAT_ORDER, self.feature_type_col_map)
                ] 
                for arr in self.grouped
            ]

    def __len__(self):
        return self._cum_examples_in_group[-1] if len(self._cum_examples_in_group) else 0

    def __getitem__(self, idx):
        g_idx = bisect(self._cum_examples_in_group, idx)
        e_idx = idx - self._cum_examples_in_group[g_idx-1] if g_idx else idx

        group =  self.grouped[g_idx]

        tensors = [
                   torch.from_numpy(feat[e_idx * self.stride:e_idx*self.stride + self.example_length])
                   if feat.size else torch.empty(0)
                   for feat in group
                  ]

        return OrderedDict(zip(FEAT_NAMES, tensors))


class TFTDataset(Dataset):
    def __init__(self, path, config):
        super(TFTDataset).__init__()
        self.features = config.features
        self.data = pd.read_csv(path, index_col=0)
        self.example_length = config.example_length
        self.stride = config.dataset_stride

        # name field is a column name.
        # there can be multiple entries with the same name because one column can be interpreted in many ways
        time_col_name = next(x.name for x in self.features if x.feature_type==InputTypes.TIME)
        id_col_name   = next(x.name for x in self.features if x.feature_type==InputTypes.ID)
        if not id_col_name in self.data.columns:
            id_col_name = DEFAULT_ID_COL
            self.features = [x for x in self.features if x.feature_type!=InputTypes.ID]
            self.features.append(FeatureSpec(DEFAULT_ID_COL, InputTypes.ID, DataTypes.CATEGORICAL))
        col_dtypes    = {v.name:DTYPE_MAP[v.feature_embed_type] for v in self.features}


        self.data.sort_values(time_col_name,inplace=True)
        self.data = self.data[set(x.name for x in self.features)] #leave only relevant columns
        self.data = self.data.astype(col_dtypes)
        self.data = self.data.groupby(id_col_name).filter(lambda group: len(group) >= self.example_length)
        self.grouped = list(self.data.groupby(id_col_name))

        self._cum_examples_in_group = np.cumsum([(len(g[1]) - self.example_length + 1)//self.stride for g in self.grouped])

    def __len__(self):
        return self._cum_examples_in_group[-1]

    def __getitem__(self, idx):
        g_idx = len([x for x in self._cum_examples_in_group if x <= idx])
        e_idx = idx - self._cum_examples_in_group[g_idx-1] if g_idx else idx

        group =  self.grouped[g_idx][1]
        sliced = group.iloc[e_idx * self.stride:e_idx*self.stride + self.example_length]

        # We need to be sure that tensors are returned in the correct order
        tensors = tuple([] for _ in range(8))
        for v in self.features:
            if v.feature_type == InputTypes.STATIC and v.feature_embed_type == DataTypes.CATEGORICAL:
                tensors[0].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.STATIC and v.feature_embed_type == DataTypes.CONTINUOUS:
                tensors[1].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.KNOWN and v.feature_embed_type == DataTypes.CATEGORICAL:
                tensors[2].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.KNOWN and v.feature_embed_type == DataTypes.CONTINUOUS:
                tensors[3].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.OBSERVED and v.feature_embed_type == DataTypes.CATEGORICAL:
                tensors[4].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.OBSERVED and v.feature_embed_type == DataTypes.CONTINUOUS:
                tensors[5].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.TARGET:
                tensors[6].append(torch.from_numpy(sliced[v.name].to_numpy()))
            elif v.feature_type == InputTypes.ID:
                tensors[7].append(torch.from_numpy(sliced[v.name].to_numpy()))


        tensors = [torch.stack(x, dim=-1) if x else torch.empty(0) for x in tensors]

        return OrderedDict(zip(FEAT_NAMES, tensors))
        
def get_dataset_splits(df, config):

    if hasattr(config, 'relative_split') and config.relative_split:
        forecast_len = config.example_length - config.encoder_length
        # The valid split is shifted from the train split by number of the forecast steps to the future.
        # The test split is shifted by the number of the forecast steps from the valid split
        train = []
        valid = []
        test = []

        for _, group in df.groupby(DEFAULT_ID_COL):
            index = group[config.time_ids]
            _train = group.loc[index < config.valid_boundary]
            _valid = group.iloc[(len(_train) - config.encoder_length):(len(_train) + forecast_len)]
            _test = group.iloc[(len(_train) - config.encoder_length + forecast_len):(len(_train) + 2*forecast_len)]
            train.append(_train)
            valid.append(_valid)
            test.append(_test)

        train = pd.concat(train, axis=0)
        valid = pd.concat(valid, axis=0)
        test = pd.concat(test, axis=0)
    else:
        index = df[config.time_ids]
        train = df.loc[(index >= config.train_range[0]) & (index < config.train_range[1])]
        valid = df.loc[(index >= config.valid_range[0]) & (index < config.valid_range[1])]
        test  = df.loc[(index >= config.test_range[0]) & (index < config.test_range[1])]

    return train, valid, test

def flatten_ids(df, config):

    if config.missing_id_strategy == 'drop':
        if hasattr(config, 'combine_ids') and config.combine_ids:
            index = np.logical_or.reduce([df[c].isna() for c in config.combine_ids])
        else:
            id_col = next(x.name for x in config.features if x.feature_type == InputTypes.ID)
            index = df[id_col].isna()
        index = index[index == True].index # Extract indices of nans
        df.drop(index, inplace=True)

    if not (hasattr(config, 'combine_ids') and config.combine_ids):
        id_col = next(x.name for x in config.features if x.feature_type == InputTypes.ID)
        ids = df[id_col].apply(str)
        df.drop(id_col, axis=1, inplace=True)
        encoder = sklearn.preprocessing.LabelEncoder().fit(ids.values)
        df[DEFAULT_ID_COL] = encoder.transform(ids)
        encoders = OrderedDict({id_col: encoder})

    else:
        encoders = {c:sklearn.preprocessing.LabelEncoder().fit(df[c].values) for c in config.combine_ids}
        encoders = OrderedDict(encoders)
        lens = [len(v.classes_) for v in encoders.values()]
        clens = np.roll(np.cumprod(lens), 1)
        clens[0] = 1

        # this takes a looooooot of time. Probably it would be better to create 2 dummy columns
        df[DEFAULT_ID_COL] = df.apply(lambda row: sum([encoders[c].transform([row[c]])[0]*clens[i] for i,c in enumerate(encoders.keys())]), axis=1)
        df.drop(config.combine_ids, axis=1, inplace=True)

    return DEFAULT_ID_COL, encoders

def impute(df, config):
    #XXX This ensures that out scaling will have the same mean. We still need to check the variance
    if not hasattr(config, 'missing_data_label'):
        return df, None
    else:
        imp = SimpleImputer(missing_values=config.missing_data_label, strategy='mean')
        mask = df.applymap(lambda x: True if x == config.missing_data_label else False)
        data = df.values
        col_mask = (data == config.missing_data_label).all(axis=0)
        data[:,~col_mask] = imp.fit_transform(data)
        return data, mask

def normalize_reals(train, valid, test, config, id_col=DEFAULT_ID_COL):
    tgt_cols = [x.name for x in config.features if x.feature_type == InputTypes.TARGET]
    real_cols = list(set(v.name for v in config.features if v.feature_embed_type == DataTypes.CONTINUOUS).difference(set(tgt_cols)))
    real_scalers = {}
    tgt_scalers = {}

    def apply_scalers(df, name=None):
        if name is None:
            name = df.name
        mask = df.applymap(lambda x: True if x == config.missing_data_label else False) if hasattr(config, 'missing_data_label') else None
        df[real_cols] = real_scalers[name].transform(df[real_cols])
        if mask is not None and any(mask):
            df[real_cols].mask(mask, 10**9)
        df[tgt_cols] = tgt_scalers[name].transform(df[tgt_cols])
        return df

    if config.scale_per_id:
        for identifier, sliced in train.groupby(id_col):
            data = sliced[real_cols]
            data, _ = impute(data, config)
            real_scalers[identifier] = sklearn.preprocessing.StandardScaler().fit(data)
            # XXX We should probably remove examples that contain NaN as a target
            target = sliced[tgt_cols]
            tgt_scalers[identifier] = sklearn.preprocessing.StandardScaler().fit(target)

        train = train.groupby(id_col).apply(apply_scalers)
        # For valid and testing leave only timeseries previously present in train subset
        # XXX for proper data science we should consider encoding unseen timeseries as a special case, not throwing them away
        valid = valid.loc[valid[id_col].isin(real_scalers.keys())]
        valid = valid.groupby(id_col).apply(apply_scalers)
        test = test.loc[test[id_col].isin(real_scalers.keys())]
        test = test.groupby(id_col).apply(apply_scalers)

    else:
        data, _ = impute(train[real_cols], config)
        real_scalers[''] = sklearn.preprocessing.StandardScaler().fit(data)
        tgt_scalers[''] = sklearn.preprocessing.StandardScaler().fit(train[tgt_cols])

        train = apply_scalers(train, name='')
        valid = apply_scalers(valid, name='')
        test = apply_scalers(test, name='')

    return train, valid, test, real_scalers, tgt_scalers

def encode_categoricals(train, valid, test, config):
    cat_encodings = {}
    cat_cols = list(set(v.name for v in config.features if v.feature_embed_type == DataTypes.CATEGORICAL and v.feature_type != InputTypes.ID))
    num_classes = [] #XXX Maybe we should modify config based on this value? Or send a warninig?
                     # For TC performance reasons we might want for num_classes[i] be divisible by 8

    # Train categorical encoders
    for c in cat_cols:
        if config.missing_cat_data_strategy == 'special_token':
            #XXX this will probably require some data augmentation
            unique = train[c].unique()
            valid[c].loc[valid[c].isin(unique)] = '<UNK>'
            test[c].loc[test[c].isin(unique)] = '<UNK>'

        if config.missing_cat_data_strategy == 'encode_all' or \
                config.missing_cat_data_strategy == 'special_token':
            srs = pd.concat([train[c], valid[c], test[c]]).apply(str)
            cat_encodings[c] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
        elif config.missing_cat_data_strategy == 'drop':
            # TODO: implement this. In addition to dropping rows this has to split specific time series in chunks
            # to prevent data from having temporal gaps
            pass
        num_classes.append(srs.nunique())
    print('Categorical variables encodings lens: ', num_classes)


    for split in [train, valid, test]:
        for c in cat_cols:
            srs = split[c].apply(str)
            split[c] = srs
            split.loc[:,c] = cat_encodings[c].transform(srs)

    return cat_encodings


def preprocess(src_path, dst_path, config):
    df = pd.read_csv(src_path, index_col=0)

    for c in config.features:
        if c.feature_embed_type == DataTypes.DATE:
            df[c.name] = pd.to_datetime(df[c.name])

    # Leave only columns relevant to preprocessing
    relevant_columns = list(set([f.name for f in config.features] + [config.time_ids]))
    df = df[relevant_columns]


    id_col, id_encoders = flatten_ids(df, config)
    df = df.reindex(sorted(df.columns), axis=1)
    
    train, valid, test = get_dataset_splits(df, config)
   
    # Length filter the data (all timeseries shorter than example len will be dropped)
    #for df in [train, valid, test]:
    #    df.groupby(id_col).filter(lambda x: len(x) >= config.example_length)
    train = pd.concat([x[1] for x in train.groupby(id_col) if len(x[1]) >= config.example_length])
    valid = pd.concat([x[1] for x in valid.groupby(id_col) if len(x[1]) >= config.example_length])
    test  = pd.concat([x[1] for x in test.groupby(id_col)  if len(x[1]) >= config.example_length])

    train, valid, test, real_scalers, tgt_scalers = normalize_reals(train, valid, test, config, id_col)

    cat_encodings = encode_categoricals(train, valid, test, config)

    os.makedirs(dst_path, exist_ok=True)
    
    train.to_csv(os.path.join(dst_path, 'train.csv'))
    valid.to_csv(os.path.join(dst_path, 'valid.csv'))
    test.to_csv(os.path.join(dst_path, 'test.csv'))

    # Save relevant columns in binary form for faster dataloading
    # IMORTANT: We always expect id to be a single column indicating the complete timeseries
    # We also expect a copy of id in form of static categorical input!!!
    col_names = [id_col] + [x.name for x in config.features if x.feature_embed_type != DataTypes.DATE and x.feature_type != InputTypes.ID]
    grouped_train = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in train.groupby(id_col)]
    grouped_valid = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in valid.groupby(id_col)]
    grouped_test  = [x[1][col_names].values.astype(np.float32).view(dtype=np.int32) for x in test.groupby(id_col)]

    pickle.dump(grouped_train, open(os.path.join(dst_path, 'train.bin'), 'wb'))
    pickle.dump(grouped_valid, open(os.path.join(dst_path, 'valid.bin'), 'wb'))
    pickle.dump(grouped_test,  open(os.path.join(dst_path, 'test.bin'), 'wb'))

    
    with open(os.path.join(dst_path, 'real_scalers.bin'), 'wb') as f:
        pickle.dump(real_scalers, f)
    with open(os.path.join(dst_path, 'tgt_scalers.bin'), 'wb') as f:
        pickle.dump(tgt_scalers, f)
    with open(os.path.join(dst_path, 'cat_encodings.bin'), 'wb') as f:
        pickle.dump(cat_encodings, f)
    with open(os.path.join(dst_path, 'id_encoders.bin'), 'wb') as f:
        pickle.dump(id_encoders, f)
    

def sample_data(dataset, num_samples):
    if num_samples < 0:
        return dataset
    else:
        return torch.utils.data.Subset(dataset, np.random.choice(np.arange(len(dataset)), size=num_samples, replace=False))

def load_dataset(args, config, collate_fn=default_collate):
    from utils import print_once
    train_split = TFTBinaryDataset(os.path.join(args.data_path, 'train.bin'), config)
    train_split = sample_data(train_split, args.sample_data[0])
    if args.distributed_world_size > 1:
        data_sampler = DistributedSampler(train_split, args.distributed_world_size, args.distributed_rank, seed=args.seed + args.distributed_rank, drop_last=True)
    else:
        data_sampler = RandomSampler(train_split)
    train_loader = DataLoader(train_split,
                              batch_size=args.batch_size,
                              num_workers=4,
                              sampler=data_sampler, 
                              collate_fn=collate_fn,
                              pin_memory=True)

    valid_split = TFTBinaryDataset(os.path.join(args.data_path, 'valid.bin'), config)
    valid_split = sample_data(valid_split, args.sample_data[1])
    if args.distributed_world_size > 1:
        data_sampler = DistributedSampler(valid_split, args.distributed_world_size, args.distributed_rank, shuffle=False, drop_last=False)
    else:
        data_sampler = None
    valid_loader = DataLoader(valid_split, 
                              batch_size=args.batch_size, 
                              sampler=data_sampler, 
                              num_workers=4, 
                              collate_fn=collate_fn,
                              pin_memory=True)

    test_split = TFTBinaryDataset(os.path.join(args.data_path, 'test.bin'), config)
    if args.distributed_world_size > 1:
        data_sampler = DistributedSampler(test_split, args.distributed_world_size, args.distributed_rank, shuffle=False, drop_last=False)
    else:
        data_sampler = None
    test_loader = DataLoader(test_split,
                             batch_size=args.batch_size, 
                             sampler=data_sampler, 
                             num_workers=4, 
                             collate_fn=collate_fn,
                             pin_memory=True)

    print_once(f'Train split length: {len(train_split)}')
    print_once(f'Valid split length: {len(valid_split)}')
    print_once(f'Test split length: {len(test_split)}')

    return train_loader, valid_loader, test_loader

def standarize_electricity(path):
    """Code taken from https://github.com/google-research/google-research/blob/master/tft/script_download_data.py"""
    df = pd.read_csv(os.path.join(path, 'LD2011_2014.txt'), index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
  
    # Used to determine the start and end dates of a series
    output = df.resample('1h').mean().replace(0., np.nan)
  
    earliest_time = output.index.min()
  
    df_list = []
    for label in output:
        print('Processing {}'.format(label))
        srs = output[label]
  
        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)
  
        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)
  
        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month
  
        df_list.append(tmp)
  
    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)
  
    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()
  
    output.to_csv(os.path.join(path, 'standarized.csv'))


def standarize_traffic(path):
    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            l = [
                variable_type(i) for i in s.replace('[', '').replace(']', '').split()
            ]
        else:
            l = [
                variable_type(i)
                for i in s.replace('[', '').replace(']', '').split(delimiter)
            ]
  
        return l
  
    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(path, filename), 'r') as dat:
            l = process_list(dat.readlines()[0])
        return l
  
    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(path, filename), 'r') as dat:
            lines = dat.readlines()
            for i, line in enumerate(lines):
                if (i + 1) % 50 == 0:
                    print('Completed {} of {} rows for {}'.format(i + 1, len(lines),
                                                                filename))
                array = [
                    process_list(row_split, variable_type=float, delimiter=None)
                    for row_split in process_list(
                        line, variable_type=str, delimiter=';')
                ]
                array_list.append(array)
  
        return array_list
  
    shuffle_order = np.array(read_single_list('randperm')) - 1  # index from 0
    train_dayofweek = read_single_list('PEMS_trainlabels')
    train_tensor = read_matrix('PEMS_train')
    test_dayofweek = read_single_list('PEMS_testlabels')
    test_tensor = read_matrix('PEMS_test')
  
    # Inverse permutate shuffle order
    print('Shuffling')
    inverse_mapping = {
        new_location: previous_location
        for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array([
        inverse_mapping[new_location]
        for new_location, _ in enumerate(shuffle_order)
    ])
  
    # Group and reoder based on permuation matrix
    print('Reodering')
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)
  
    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]
  
    # Put everything back into a dataframe
    print('Parsing as dataframe')
    labels = ['traj_{}'.format(i) for i in read_single_list('stations_list')]
  
    hourly_list = []
    for day, day_matrix in enumerate(combined_tensor):
        # Hourly data
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        hourly['hour_on_day'] = [int(i / 6) for i in hourly.index
                                ]  # sampled at 10 min intervals
        if hourly['hour_on_day'].max() > 23 or hourly['hour_on_day'].min() < 0:
            raise ValueError('Invalid hour! {}-{}'.format(
                hourly['hour_on_day'].min(), hourly['hour_on_day'].max()))
  
        hourly = hourly.groupby('hour_on_day', as_index=True).mean()[labels]
        hourly['sensor_day'] = day
        hourly['time_on_day'] = hourly.index
        hourly['day_of_week'] = day_of_week[day]
  
        hourly_list.append(hourly)
  
    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)
  
    # Flatten such that each entitiy uses one row in dataframe
    store_columns = [c for c in hourly_frame.columns if 'traj' in c]
    other_columns = [c for c in hourly_frame.columns if 'traj' not in c]
    flat_df = pd.DataFrame(columns=['values', 'prev_values', 'next_values'] +
                           other_columns + ['id'])
  
    for store in store_columns:
        print('Processing {}'.format(store))
  
        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ['values'] + other_columns
        sliced['id'] = int(store.replace('traj_', ''))
  
        # Sort by Sensor-date-time
        key = sliced['id'].apply(str) \
                + sliced['sensor_day'].apply(lambda x: '_{:03d}'.format(x)) \
                + sliced['time_on_day'].apply(lambda x: '_{:03d}'.format(x))
        sliced = sliced.set_index(key).sort_index()
  
        sliced['values'] = sliced['values'].fillna(method='ffill')
        sliced['prev_values'] = sliced['values'].shift(1)
        sliced['next_values'] = sliced['values'].shift(-1)
  
        flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)
  
    # Filter to match range used by other academic papers
    index = flat_df['sensor_day']
    flat_df = flat_df[index < 173].copy()
  
    # Creating columns fo categorical inputs
    flat_df['categorical_id'] = flat_df['id'].copy()
    flat_df['hours_from_start'] = flat_df['time_on_day'] \
        + flat_df['sensor_day']*24.
    flat_df['categorical_day_of_week'] = flat_df['day_of_week'].copy()
    flat_df['categorical_time_on_day'] = flat_df['time_on_day'].copy()
  
    flat_df.to_csv(os.path.join(path, 'standarized.csv'))

