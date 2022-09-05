from data.feature_spec import FeatureSpec
from data.outbrain.defaults import ONEHOT_CHANNEL, MULTIHOT_CHANNEL
from argparse import ArgumentParser
import random
import json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--feature_spec_in', type=str, default='feature_spec.yaml',
                        help='Name of the input feature specification file')
    parser.add_argument('--output', type=str)
    parser.add_argument('--max_size', type=int, default=256,
                        help='Max embedding size to pick')
    return parser.parse_args()


def main():
    #this generator supports the following feature types:
    #onehot categorical
    #numerical
    #label
    #multihot categorical
    args = parse_args()
    fspec_in = FeatureSpec.from_yaml(args.feature_spec_in)
    max_size = args.max_size
    onehot_features = fspec_in.get_names_by_channel(ONEHOT_CHANNEL)

    multihot_features = fspec_in.get_names_by_channel(MULTIHOT_CHANNEL)
    sizes = {feature: random.randint(1,max_size) for feature in onehot_features+multihot_features}
    with open(args.output, "w") as opened:
        json.dump(sizes, opened)

if __name__ == "__main__":
    main()