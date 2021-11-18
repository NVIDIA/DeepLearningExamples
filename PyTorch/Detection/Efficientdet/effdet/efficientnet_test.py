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

import torch

from efficientnet import EfficientNet, efficientnet_configs

def test_feature_type(net, images):
    output, features = net(images, features_only=True)
    print("[ ... Test Type ... ] Type of output {} features {}".format(type(output), type(features)))

def test_feature_dimensions(net, images):
    output, features = net(images, features_only=True)
    print("[ ... Test dimension ... ] Dim of output {} features {}".format(output.size(), len(features)))
    for i, x in enumerate(features):
        print("[ ... Test dimension ... ] Index {} features size {}".format(i, features[i].size()))

def test_feature_info(net, images):
    feature_info = net.feature_info
    for i, f in enumerate(feature_info):
        print("[ ... Test Feature Info ... ] Index {} features info {}".format(i, f))

def main():
    global_config = efficientnet_configs['fanout']
    net = EfficientNet(width_coeff=1, depth_coeff=1, dropout=0.2, num_classes=1000, global_config=global_config, out_indices=[2,3,4])
    images = torch.rand((2, 3, 512, 512))
    test_feature_type(net, images)
    test_feature_dimensions(net, images)
    test_feature_info(net, images)
    print("Model Layer Names")
    for n, m in net.named_modules():
        print(n)

if __name__ == '__main__':
    main()