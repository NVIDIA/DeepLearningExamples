# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from argparse import ArgumentParser
import os
import torch
import pandas as pd

from feature_spec import FeatureSpec
from neumf_constants import USER_CHANNEL_NAME, ITEM_CHANNEL_NAME, LABEL_CHANNEL_NAME


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='',
                        help='Path to input data directory')
    parser.add_argument('--feature_spec_in', type=str, default='feature_spec.yaml',
                        help='Name of the input feature specification file, or path relative to data directory.')
    parser.add_argument('--output', type=str, default='/data',
                        help='Path to output data directory')
    parser.add_argument('--feature_spec_out', type=str, default='feature_spec.yaml',
                        help='Name of the output feature specification file, or path relative to data directory.')
    return parser.parse_args()


def main():
    args = parse_args()
    args_output = args.output
    args_path = args.path
    args_feature_spec_in = args.feature_spec_in
    args_feature_spec_out = args.feature_spec_out

    feature_spec_path = os.path.join(args_path, args_feature_spec_in)
    feature_spec = FeatureSpec.from_yaml(feature_spec_path)

    # Only three features are transcoded - this is NCF specific
    user_feature_name = feature_spec.channel_spec[USER_CHANNEL_NAME][0]
    item_feature_name = feature_spec.channel_spec[ITEM_CHANNEL_NAME][0]
    label_feature_name = feature_spec.channel_spec[LABEL_CHANNEL_NAME][0]

    categorical_features = [user_feature_name, item_feature_name]

    found_cardinalities = {f: 0 for f in categorical_features}

    new_source_spec = {}
    for mapping_name, mapping in feature_spec.source_spec.items():
        # Load all chunks and link into one df
        chunk_dfs = []
        for chunk in mapping:
            assert chunk['type'] == 'csv', "Only csv files supported in this transcoder"
            file_dfs = []
            for file in chunk['files']:
                path_to_load = os.path.join(feature_spec.base_directory, file)
                file_dfs.append(pd.read_csv(path_to_load, header=None))
            chunk_df = pd.concat(file_dfs, ignore_index=True)
            chunk_df.columns = chunk['features']
            chunk_df.reset_index(drop=True, inplace=True)
            chunk_dfs.append(chunk_df)
        mapping_df = pd.concat(chunk_dfs, axis=1)  # This takes care of making sure feature names are unique

        for feature in categorical_features:
            mapping_cardinality = mapping_df[feature].max() + 1
            previous_cardinality = found_cardinalities[feature]
            found_cardinalities[feature] = max(previous_cardinality, mapping_cardinality)

        # We group together users and items, while separating labels. This is because of the target dtypes: ids are int,
        # while labels are float to compute loss.
        ints_tensor = torch.from_numpy(mapping_df[[user_feature_name, item_feature_name]].values).long()
        ints_file = f"{mapping_name}_data_0.pt"
        ints_chunk = {"type": "torch_tensor",
                      "features": [user_feature_name, item_feature_name],
                      "files": [ints_file]}
        torch.save(ints_tensor, os.path.join(args_output, ints_file))

        floats_tensor = torch.from_numpy(mapping_df[[label_feature_name]].values).float()
        floats_file = f"{mapping_name}_data_1.pt"
        floats_chunk = {"type": "torch_tensor",
                        "features": [label_feature_name],
                        "files": [floats_file]}
        torch.save(floats_tensor, os.path.join(args_output, floats_file))

        new_source_spec[mapping_name] = [ints_chunk, floats_chunk]

    for feature in categorical_features:
        found_cardinality = found_cardinalities[feature]
        declared_cardinality = feature_spec.feature_spec[feature].get('cardinality', 'auto')
        if declared_cardinality != "auto":
            declared = int(declared_cardinality)
            assert declared >= found_cardinality, "Specified cardinality conflicts data"
            found_cardinalities[feature] = declared

    new_inner_feature_spec = {
        user_feature_name: {
            "dtype": "torch.int64",
            "cardinality": int(found_cardinalities[user_feature_name])
        },
        item_feature_name: {
            "dtype": "torch.int64",
            "cardinality": int(found_cardinalities[item_feature_name])
        },
        label_feature_name: {
            "dtype": "torch.float32"
        }
    }

    new_feature_spec = FeatureSpec(feature_spec=new_inner_feature_spec,
                                   source_spec=new_source_spec,
                                   channel_spec=feature_spec.channel_spec,
                                   metadata=feature_spec.metadata,
                                   base_directory="")
    feature_spec_save_path = os.path.join(args_output, args_feature_spec_out)
    new_feature_spec.to_yaml(output_path=feature_spec_save_path)


if __name__ == '__main__':
    main()
