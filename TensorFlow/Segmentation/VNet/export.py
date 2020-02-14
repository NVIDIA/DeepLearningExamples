# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import argparse

import tensorflow as tf

from utils.data_loader import MSDDataset
from utils.model_fn import vnet_v2
from utils.tf_export import to_savedmodel, to_tf_trt, to_onnx

PARSER = argparse.ArgumentParser(description="V-Net")

PARSER.add_argument('--to', dest='to', choices=['savedmodel', 'tftrt', 'onnx'], required=True)

PARSER.add_argument('--use_amp', dest='use_amp', action='store_true', default=False)
PARSER.add_argument('--use_xla', dest='use_xla', action='store_true', default=False)
PARSER.add_argument('--compress', dest='compress', action='store_true', default=False)

PARSER.add_argument('--input_shape',
                    nargs='+',
                    type=int,
                    help="""Model's input shape""")

PARSER.add_argument('--data_dir',
                    type=str,
                    help="""Directory where the dataset is located""")

PARSER.add_argument('--checkpoint_dir',
                    type=str,
                    help="""Directory where the checkpoint is located""")

PARSER.add_argument('--savedmodel_dir',
                    type=str,
                    help="""Directory where the savedModel is located""")

PARSER.add_argument('--precision',
                    type=str,
                    choices=['FP32', 'FP16', 'INT8'],
                    help="""Precision for the model""")


def main():
    """
    Starting point of the application
    """
    flags = PARSER.parse_args()

    if flags.to == 'savedmodel':
        params = {
            'labels': ['0', '1', '2'],
            'batch_size': 1,
            'input_shape': flags.input_shape,
            'convolution_size': 3,
            'downscale_blocks': [3, 3, 3],
            'upscale_blocks': [3, 3],
            'upsampling': 'transposed_conv',
            'pooling': 'conv_pool',
            'normalization_layer': 'batchnorm',
            'activation': 'relu'
        }
        to_savedmodel(input_shape=flags.input_shape,
                      model_fn=vnet_v2,
                      checkpoint_dir=flags.checkpoint_dir,
                      output_dir='./saved_model',
                      input_names=['IteratorGetNext'],
                      output_names=['vnet/loss/total_loss_ref'],
                      use_amp=flags.use_amp,
                      use_xla=flags.use_xla,
                      compress=flags.compress,
                      params=argparse.Namespace(**params))
    if flags.to == 'tftrt':
        ds = MSDDataset(json_path=flags.data_dir + "/dataset.json",
                        interpolator='linear')
        iterator = ds.test_fn(count=1).make_one_shot_iterator()
        features = iterator.get_next()

        sess = tf.Session()

        def input_data():
            return {'input_tensor:0': sess.run(features)}

        to_tf_trt(savedmodel_dir=flags.savedmodel_dir,
                  output_dir='./tf_trt_model',
                  precision=flags.precision,
                  feed_dict_fn=input_data,
                  num_runs=1,
                  output_tensor_names=['vnet/Softmax:0'],
                  compress=flags.compress)
    if flags.to == 'onnx':
        raise NotImplementedError('Currently ONNX not supported for 3D models')


if __name__ == '__main__':
    main()
