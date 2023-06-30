# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import yaml
import argparse


variants = dict(
    # Generates 16 GiB embedding tables
    criteo_t15_synthetic=dict(
        num_numerical=13,
        cardinalities=[7912889, 33823, 17139, 7339, 20046, 4, 7105, 1382, 63, 5554114, 582469, 245828, 11, 2209,
                       10667, 104, 4, 968, 15, 8165896, 2675940, 7156453, 302516, 12022, 97, 35],
        hotness=26 * [1],
        alpha=26 * [1.45]
    ),
    # Generates 85 GiB embedding tables
    criteo_t3_synthetic=dict(
        num_numerical=13,
        cardinalities=[45833188,36747,1572176,345139,11,2209,11268,128,4,975,15,48937457,17246,11316796,40094537,
                       452104,12607,105,36,7414,20244,4,7115,1442,63,29275261],
        hotness=26 * [1],
        alpha=26 * [1.45]
    ),
    # Generates 421 GiB
    criteo_t0_synthetic=dict(
        num_numerical=13,
        cardinalities=[227605432, 39061, 3067956, 405283, 11, 2209, 11939, 155, 4, 977, 15, 292775614, 17296,
                       40790948, 187188510, 590152, 12974, 109, 37, 7425, 20266, 4, 7123, 1544, 64, 130229467],
        hotness=26 * [1],
        alpha=26 * [1.45]
    ),
)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic feature spec")
    parser.add_argument('--dst', default='feature_spec.yaml', type=str, help='Output path')
    parser.add_argument('--variant', choices=list(variants.keys()), required=True, type=str,
                        help='Variant of the synthetic dataset to be used')
    args = parser.parse_args()
    num_numerical, cardinalities, hotness, alphas = tuple(variants[args.variant].values())

    feature_spec = {}
    for i, (c, h, a) in enumerate(zip(cardinalities, hotness, alphas)):
        name = f'cat_{i}'
        f = dict(cardinality=c, hotness=h, alpha=a, dtype='int32')
        feature_spec[name] = f

    for i in range(num_numerical):
        name = f'num_{i}'
        feature_spec[name] = dict(dtype='float16')

    feature_spec['label'] = dict(dtype='int8')

    channel_spec = {}
    channel_spec['categorical'] = [k for k in feature_spec.keys() if 'cat' in k]
    channel_spec['numerical'] = [k for k in feature_spec.keys() if 'num' in k]
    channel_spec['label'] = ['label']

    source_spec = None
    full_spec = dict(feature_spec=feature_spec, channel_spec=channel_spec, source_spec=source_spec)

    with open(args.dst, 'w') as f:
        yaml.dump(data=full_spec, stream=f)


if __name__ == '__main__':
    main()
