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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.feature_spec import FeatureSpec, FEATURES_SELECTOR, TYPE_SELECTOR, FILES_SELECTOR
from data.outbrain.defaults import MULTIHOT_CHANNEL, PARQUET_TYPE


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
    parser.add_argument('--chunk_size', type=int, default=65536,
                        help='Number of rows to write out per partition')
    parser.add_argument('--minimum_partition_number', type=int, default=8,
                        help='throw error if each mapping does not produce at least this many partitions')
    return parser.parse_args()

def check_only_one_file_per_chunk(feature_spec):
    for mapping in feature_spec.source_spec.values():
        for chunk in mapping:
            chunk_files = chunk[FILES_SELECTOR]
            assert len(chunk_files) == 1
            assert chunk[TYPE_SELECTOR] == 'csv'

def main():
    args = parse_args()
    args_output = args.output
    args_input = args.input
    args_feature_spec_in = args.feature_spec_in
    args_feature_spec_out = args.feature_spec_out
    batch_size = args.chunk_size

    fspec_in_path = os.path.join(args_input, args_feature_spec_in)
    fspec_in = FeatureSpec.from_yaml(fspec_in_path)

    os.makedirs(args.output, exist_ok=True)

    paths_per_mapping = dict()

    check_only_one_file_per_chunk(fspec_in)

    for mapping_name, mapping in fspec_in.source_spec.items():

        paths_per_mapping[mapping_name]=[]
        df_iterators = []
        for chunk in mapping:
            # We checked earlier it's a single file chunk
            path_to_load = os.path.join(fspec_in.base_directory, chunk[FILES_SELECTOR][0])
            chunk_iterator = pd.read_csv(path_to_load, header=None, chunksize=batch_size, names=chunk[FEATURES_SELECTOR])
            df_iterators.append(chunk_iterator)

        zipped = zip(*df_iterators)
        # writer = None
        for chunk_id, chunks in enumerate(zipped):
            # chunks is now a list of the chunk_id-th segment of each dataframe iterator and contains all columns
            mapping_df = pd.concat(chunks, axis=1)  # This takes care of making sure feature names are unique

            #transform multihots from strings to objects # TODO: find a better way to do this
            multihot_features = fspec_in.get_names_by_channel(MULTIHOT_CHANNEL)
            for feature in multihot_features:
                mapping_df[feature] = mapping_df[feature].apply(lambda x: np.fromstring(x[1:-1], sep=' ,'))

            # prepare path
            partition_path = f"{mapping_name}_{chunk_id}.parquet"
            paths_per_mapping[mapping_name].append(partition_path)
            partition_path_abs = os.path.join(args.output, partition_path)

            #write to parquet
            mapping_table = pa.Table.from_pandas(mapping_df)
            pq.write_table(mapping_table, partition_path_abs)

    # Prepare the new feature spec
    new_source_spec = {}
    old_source_spec = fspec_in.source_spec
    for mapping_name in old_source_spec.keys():
        #check if we met the required partitions number
        min_partitions = args.minimum_partition_number
        got_partitions = len(paths_per_mapping[mapping_name])
        assert got_partitions>min_partitions, f"Not enough partitions generated for mapping:{mapping_name}. Expected at least {min_partitions}, got {got_partitions}"

        all_features = []
        for chunk in old_source_spec[mapping_name]:
            all_features = all_features + chunk[FEATURES_SELECTOR]

        new_source_spec[mapping_name] = []
        new_source_spec[mapping_name].append({TYPE_SELECTOR: PARQUET_TYPE,
                                          FEATURES_SELECTOR: all_features,
                                          FILES_SELECTOR: paths_per_mapping[mapping_name]})

    fspec_out = FeatureSpec(feature_spec=fspec_in.feature_spec, source_spec=new_source_spec,
                            channel_spec=fspec_in.channel_spec, metadata=fspec_in.metadata)
    fspec_out.base_directory = args.output

    feature_spec_save_path = os.path.join(args_output, args_feature_spec_out)
    fspec_out.to_yaml(output_path=feature_spec_save_path)


if __name__ == '__main__':
    main()