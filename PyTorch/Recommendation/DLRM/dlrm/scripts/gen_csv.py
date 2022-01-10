from dlrm.data.defaults import NUMERICAL_CHANNEL, LABEL_CHANNEL
from dlrm.data.feature_spec import FeatureSpec
from argparse import ArgumentParser
import pandas as pd
import os
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--feature_spec_in', type=str, default='feature_spec.yaml',
                        help='Name of the input feature specification file')
    parser.add_argument('--output', type=str, default='/data')
    parser.add_argument('--size', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_size = args.size
    fspec_in = FeatureSpec.from_yaml(args.feature_spec_in)
    fspec_in.base_directory = args.output
    cat_cardinalities = fspec_in.get_categorical_sizes()
    cat_names = fspec_in.get_categorical_feature_names()
    cardinalities = {name: cardinality for name, cardinality in zip(cat_names, cat_cardinalities)}
    input_label_feature_name = fspec_in.channel_spec[LABEL_CHANNEL][0]
    numerical_names_set = set(fspec_in.channel_spec[NUMERICAL_CHANNEL])
    for mapping_name, mapping in fspec_in.source_spec.items():
        for chunk in mapping:
            assert chunk['type'] == 'csv', "Only csv files supported in this generator"
            assert len(chunk['files']) == 1, "Only one file per chunk supported in this transcoder"
            path_to_save = os.path.join(fspec_in.base_directory, chunk['files'][0])
            data = []
            for name in chunk['features']:
                if name == input_label_feature_name:
                    data.append(np.random.randint(0, 1, size=dataset_size))
                elif name in numerical_names_set:
                    data.append(np.random.rand(dataset_size))
                else:
                    local_cardinality = cardinalities[name]
                    data.append(np.random.randint(0, local_cardinality, size=dataset_size))
            values = np.stack(data).T
            to_save = pd.DataFrame(values, columns=chunk['features'])
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            to_save.to_csv(path_to_save, index=False, header=False)


if __name__ == "__main__":
    main()
