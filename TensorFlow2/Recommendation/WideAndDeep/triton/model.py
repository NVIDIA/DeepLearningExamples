# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from typing import List

import tensorflow as tf

from data.outbrain.features import get_outbrain_feature_spec, EMBEDDING_DIMENSIONS
from trainer.model.widedeep import wide_deep_model


def update_argparser(parser):
    parser.add_argument('--deep-hidden-units', type=int, default=[1024, 1024, 1024, 1024, 1024], nargs='+',
                        help='Hidden units per layer for deep model, separated by spaces')

    parser.add_argument('--deep-dropout', type=float, default=0.1,
                        help='Dropout regularization for deep model')

    parser.add_argument('--combiner', type=str, default='sum', choices=['mean', 'sum'],
                        help='Type of aggregation used for multi hot categorical features')

    parser.add_argument('--precision', type=str, default="fp16", choices=['fp32', 'fp16'],
                        help='Precision of the ops. AMP will be used in case of fp16')

    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Path to directory containing checkpoint')

def get_model(
        *,
        deep_hidden_units: List[int],
        deep_dropout: float,
        combiner: str,
        checkpoint_dir: str,
        precision: str = "fp32",
        batch_size: int = 131072
):
    args = {
        'deep_hidden_units': deep_hidden_units,
        'deep_dropout': deep_dropout,
        'combiner': combiner
    }

    args = SimpleNamespace(**args)

    #This will be changed in the future when feature spec support for triton is added
    feature_spec = get_outbrain_feature_spec("")
    embedding_dimensions = EMBEDDING_DIMENSIONS
    model, features = wide_deep_model(args, feature_spec, embedding_dimensions)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    inputs = features.values()
    outputs = model(features, training=False)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    @tf.function
    def call_fn(*model_inputs):
        return model(model_inputs, training=False)

    return model, call_fn


if __name__ == '__main__':
    get_model(deep_hidden_units=[1024, 1024, 1024, 1024, 1024], deep_dropout=0.1, combiner='sum',
              checkpoint_dir='/tmp/wd2/checkpoint')
