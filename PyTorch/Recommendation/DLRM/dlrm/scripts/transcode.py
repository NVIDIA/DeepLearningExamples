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
from collections import defaultdict

import torch
import pandas as pd

from dlrm.data.feature_spec import FeatureSpec
from dlrm.data.defaults import CATEGORICAL_CHANNEL, NUMERICAL_CHANNEL, LABEL_CHANNEL, CARDINALITY_SELECTOR
from dlrm.data.defaults import get_categorical_feature_type


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='',
                        help='Path to input data directory')
    parser.add_argument('--feature_spec_in', type=str, default='feature_spec.yaml',
                        help='Name of the input feature specification file')
    parser.add_argument('--output', type=str, default='/data',
                        help='Path to output data directory')
    parser.add_argument('--feature_spec_out', type=str, default='feature_spec.yaml',
                        help='Name of the output feature specification file')
    parser.add_argument('--chunk_size', type=int, default=65536)
    return parser.parse_args()


def main():
    args = parse_args()
    args_output = args.output
    args_input = args.input
    args_feature_spec_in = args.feature_spec_in
    args_feature_spec_out = args.feature_spec_out
    batch_size = args.chunk_size

    fspec_in_path = os.path.join(args_input, args_feature_spec_in)
    fspec_in = FeatureSpec.from_yaml(fspec_in_path)

    input_label_feature_name = fspec_in.channel_spec[LABEL_CHANNEL][0]
    input_numerical_features_list = fspec_in.channel_spec[NUMERICAL_CHANNEL]
    input_categorical_features_list = fspec_in.channel_spec[CATEGORICAL_CHANNEL]

    # Do a pass to establish the cardinalities: they influence the type we save the dataset as
    found_cardinalities = defaultdict(lambda: 0)
    for mapping_name, mapping in fspec_in.source_spec.items():
        df_iterators = []
        for chunk in mapping:
            assert chunk['type'] == 'csv', "Only csv files supported in this transcoder"
            assert len(chunk['files']) == 1, "Only one file per chunk supported in this transcoder"
            path_to_load = os.path.join(fspec_in.base_directory, chunk['files'][0])
            chunk_iterator = pd.read_csv(path_to_load, header=None, chunksize=batch_size, names=chunk['features'])
            df_iterators.append(chunk_iterator)

        zipped = zip(*df_iterators)
        for chunks in zipped:
            mapping_df = pd.concat(chunks, axis=1)
            for feature in input_categorical_features_list:
                mapping_cardinality = mapping_df[feature].max() + 1
                previous_cardinality = found_cardinalities[feature]
                found_cardinalities[feature] = max(previous_cardinality, mapping_cardinality)

    for feature in input_categorical_features_list:
        declared_cardinality = fspec_in.feature_spec[feature][CARDINALITY_SELECTOR]
        if declared_cardinality == 'auto':
            pass
        else:
            assert int(declared_cardinality) >= found_cardinalities[feature]
            found_cardinalities[feature] = int(declared_cardinality)

    categorical_cardinalities = [found_cardinalities[f] for f in input_categorical_features_list]
    number_of_numerical_features = fspec_in.get_number_of_numerical_features()

    fspec_out = FeatureSpec.get_default_feature_spec(number_of_numerical_features=number_of_numerical_features,
                                                     categorical_feature_cardinalities=categorical_cardinalities)
    fspec_out.base_directory = args.output

    for mapping_name, mapping in fspec_in.source_spec.items():

        # open files for outputting
        label_path, numerical_path, categorical_paths = fspec_out.get_mapping_paths(mapping_name)
        for path in [label_path, numerical_path, *categorical_paths.values()]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        output_categorical_features_list = fspec_out.get_categorical_feature_names()
        numerical_f = open(numerical_path, "ab+")
        label_f = open(label_path, "ab+")
        categorical_fs = [open(categorical_paths[name], "ab+") for name in output_categorical_features_list]
        categorical_feature_types = [get_categorical_feature_type(card) for card in categorical_cardinalities]

        df_iterators = []
        for chunk in mapping:
            # We checked earlier it's a single file chunk
            path_to_load = os.path.join(fspec_in.base_directory, chunk['files'][0])
            chunk_iterator = pd.read_csv(path_to_load, header=None, chunksize=batch_size, names=chunk['features'])
            df_iterators.append(chunk_iterator)

        zipped = zip(*df_iterators)
        for chunks in zipped:
            mapping_df = pd.concat(chunks, axis=1)  # This takes care of making sure feature names are unique

            # Choose the right columns
            numerical_df = mapping_df[input_numerical_features_list]
            categorical_df = mapping_df[input_categorical_features_list]
            label_df = mapping_df[[input_label_feature_name]]

            numerical = torch.tensor(numerical_df.values)
            label = torch.tensor(label_df.values)
            categorical = torch.tensor(categorical_df.values)

            # Append them to the binary files
            numerical_f.write(numerical.to(torch.float16).cpu().numpy().tobytes())
            label_f.write(label.to(torch.bool).cpu().numpy().tobytes())
            for cat_idx, cat_feature_type in enumerate(categorical_feature_types):
                categorical_fs[cat_idx].write(
                    categorical[:, cat_idx].cpu().numpy().astype(cat_feature_type).tobytes())

    feature_spec_save_path = os.path.join(args_output, args_feature_spec_out)
    fspec_out.to_yaml(output_path=feature_spec_save_path)


if __name__ == '__main__':
    main()
