from data.feature_spec import FeatureSpec
from data.outbrain.defaults import ONEHOT_CHANNEL, MULTIHOT_CHANNEL, LABEL_CHANNEL, NUMERICAL_CHANNEL, \
    MAP_FEATURE_CHANNEL
from argparse import ArgumentParser
import pandas as pd
import os
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--feature_spec_in', type=str, default='feature_spec.yaml',
                        help='Name of the input feature specification file')
    parser.add_argument('--output', type=str, default='/data')
    parser.add_argument('--size', type=int, default=1000,
                        help='The desired number of rows in the output csv file')
    return parser.parse_args()


def main():
    #this generator supports the following feature types:
    #onehot categorical
    #numerical
    #label
    #multihot categorical
    args = parse_args()
    dataset_size = args.size
    fspec_in = FeatureSpec.from_yaml(args.feature_spec_in)

    fspec_in.base_directory = args.output

    #prepare shapes for one-hot categorical features
    onehot_features = fspec_in.get_names_by_channel(ONEHOT_CHANNEL)
    onehot_cardinalities: dict = fspec_in.get_cardinalities(onehot_features)

    multihot_features = fspec_in.get_names_by_channel(MULTIHOT_CHANNEL)
    multihot_cardinalities: dict = fspec_in.get_cardinalities(multihot_features)
    multihot_hotnesses: dict = fspec_in.get_multihot_hotnesses(multihot_features)

    input_label_feature_name = fspec_in.get_names_by_channel(LABEL_CHANNEL)[0]
    numerical_names_set = set(fspec_in.get_names_by_channel(NUMERICAL_CHANNEL))

    map_channel_features = fspec_in.get_names_by_channel(MAP_FEATURE_CHANNEL)
    map_feature = None
    if len(map_channel_features)>0:
        map_feature=map_channel_features[0]

    for mapping_name, mapping in fspec_in.source_spec.items():
        for chunk in mapping:
            assert chunk['type'] == 'csv', "Only csv files supported in this generator"
            assert len(chunk['files']) == 1, "Only one file per chunk supported in this generator"
            path_to_save = os.path.join(fspec_in.base_directory, chunk['files'][0])
            data = {}
            for name in chunk['features']:
                if name == input_label_feature_name:
                    data[name]=np.random.randint(0, 2, size=dataset_size)
                elif name in numerical_names_set:
                    data[name]=np.random.rand(dataset_size)
                elif name in set(onehot_features):
                    local_cardinality = onehot_cardinalities[name]
                    data[name]=np.random.randint(0, local_cardinality, size=dataset_size)
                elif name in set(multihot_features):
                    local_cardinality = multihot_cardinalities[name]
                    local_hotness = multihot_hotnesses[name]
                    data[name]=np.random.randint(0, local_cardinality, size=(dataset_size, local_hotness)).tolist()
                elif name == map_feature:
                    raise NotImplementedError("Cannot generate datasets with map feature enabled")
                    # TODO add a parameter that specifies max repeats and generate
                else:
                    raise ValueError(f"Cannot generate for unused features. Unknown feature: {name}")

            # Columns in the csv appear in the order they are listed in the source spec for a given chunk
            column_order = chunk['files']
            df = pd.DataFrame(data)

            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            df.to_csv(path_to_save, columns=column_order, index=False, header=False)


if __name__ == "__main__":
    main()
