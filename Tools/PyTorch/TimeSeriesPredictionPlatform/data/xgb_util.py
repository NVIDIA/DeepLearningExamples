# Copyright 2022 NVIDIA Corporation

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

def load_xgb_df(dest_path, features, ds_type):
    '''
    Loads and does some light preprocessing on the train, valid and test.
    First the csvs are read for each, then the features not present in the feature spec are dropped,
    and finally the features with datatype as object are dropped.  The final step is to prevent issues with
    xgboost training and cuDF casting.
    '''
    path = dest_path
    if not isinstance(path, pd.DataFrame):
        df = pd.read_csv(os.path.join(path, f"{ds_type}.csv"))
    else:
        df = path
    all_features = [f.name for f in features] + ['_id_']
    all_read = df.columns
    to_drop = [c for c in all_read if c not in all_features]
    df.drop(columns=to_drop, inplace=True)

    object_columns = [c for c, d in zip(df.columns, df.dtypes) if d == "object"]
    df.drop(columns=object_columns, inplace=True)

    return df

def xgb_multiID_preprocess(df, features, time_series_count):
    date = [feature.name for feature in features if feature.feature_type == "TIME"][0]
    target = [feature.name for feature in features if feature.feature_type == "TARGET"][0]
    time_series_count = time_series_count
    target_values = []
    for _, g in df.groupby("_id_"):
        target_values.append(g[[date, target]])
    final = target_values[0]
    final.rename(columns={target: f'{target}_{0}'}, inplace=True)
    for i in range(1, time_series_count):
        target_values[i].rename(columns={target: f'{target}_{i}'}, inplace=True)
        final = final.merge(target_values[i], on=date, how='outer')
    
    df = df.merge(final, on=date, how='outer')
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