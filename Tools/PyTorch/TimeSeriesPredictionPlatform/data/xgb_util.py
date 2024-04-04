# Copyright 2022-2024 NVIDIA Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import os

def select_per_group(df, start, end):
    '''
    Groups the dataframe by the _id_ and grabs elements on the slice start to end.  The resulting array
    is concat to a dataframe.
    '''
    result = []
    for _, g in df.groupby("_id_"):
        result.append(g[start:end])
    return pd.concat((result))

def select_test_group(df, encoder, example):
    '''
    Purpose of the function is to create the dataframe to pass to the xgboost predict.  After grouping by
    the _id_, each group has elements selected such that all complete time-series are chosen.  
    '''
    final = []
    for _, g in df.groupby("_id_"):
        final.append(g[encoder-1: encoder + len(g) - example])
    return  pd.concat((final))

def xgb_multiID_preprocess(df, time_feat, target_feat):
    target_values = []

    for i, g in df.groupby("_id_"):
        d = g[[time_feat, target_feat]]
        d.rename(columns={target_feat: f'{target_feat}_{i}'}, inplace=True)
        target_values.append(d)

    # faster than calling functools.reduce
    final = target_values[0]
    for t in target_values[1:]:
        final = final.merge(t, on=time_feat, how='outer')
    
    df = df.merge(final, on=time_feat, how='outer')
    return df

def feat_adder(df, lag_feats, rolling_feats):
    '''
    Main data preprocessing function for xgboost.  lag_feats and rolling_feats are both
    dictionaries from features to lists.  After grouping by the _id_
    we iterate through the lag features and move down the features i steps in the history.
    Similarly the rolling_feats are iterated through and the moving average of the past i time steps
    of that feature is added as a feature.  The names of the new lag features are the
    {feature_name}_{i}_lag and of the new rolling features are {feature_name}_{i}_rolling.
    '''
    final = []
    for _, g in df.groupby("_id_"):
        for f, v in lag_feats.items():
            for i in v:
                g['{}_{}_lag'.format(f, i)] = g[f].shift(i)
        for f, v in rolling_feats.items():
            for i in v:
                g['{}_{}_rolling'.format(f, i)] = g[f].rolling(i).sum()
        final.append(g)
    return pd.concat((final))

def data_label_split(df, target):
    '''
    Drops rows with NaN as the target value.  In addition separates the labels from
    the data by doing an inplace drop.
    '''
    df.dropna(subset=target, inplace=True)
    labels = df[target]
    df.drop(target, 1, inplace=True)
    return labels


def target_shift(df, target, feat, i):
    '''
    Brings features up that are (i+1) time steps in the future.  Currently these features are
    the target and the known/static variables.  This future target is the value that will be predicted in the trainer.
    These features have an _target added to their name.
    '''
    in_feat = target + feat
    out_feat = [f'{i}_target' for i in in_feat]
    df[out_feat] = df.groupby("_id_")[in_feat].shift(-1 * (i+1))
    return df
