# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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


from absl import app, flags


def define_dcnv2_specific_flags():
    flags.DEFINE_integer("batch_size", default=64 * 1024, help="Batch size used for training")
    flags.DEFINE_integer("valid_batch_size", default=64 * 1024, help="Batch size used for validation")
    flags.DEFINE_list("top_mlp_dims", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
    flags.DEFINE_list("bottom_mlp_dims", [512, 256, 128], "Linear layer sizes for the bottom MLP")
    flags.DEFINE_string("embedding_dim", default='128', help='Number of columns in the embedding tables')
    flags.DEFINE_enum("optimizer", default="adam", enum_values=['sgd', 'adam'],
                      help='The optimization algorithm to be used.')
    flags.DEFINE_enum("interaction", default="cross", enum_values=["dot_custom_cuda", "dot_tensorflow", "cross"],
                      help="Feature interaction implementation to use")
    flags.DEFINE_float("learning_rate", default=0.0001, help="Learning rate")
    flags.DEFINE_float("beta1", default=0.9, help="Beta1 for the Adam optimizer")
    flags.DEFINE_float("beta2", default=0.999, help="Bea2 for the Adam optimizer")
    flags.DEFINE_integer("warmup_steps", default=100,
                        help='Number of steps over which to linearly increase the LR at the beginning')
    flags.DEFINE_integer("decay_start_step", default=48000, help='Optimization step at which to start the poly LR decay')
    flags.DEFINE_integer("decay_steps", default=24000, help='Number of steps over which to decay from base LR to 0')

    flags.DEFINE_integer("num_cross_layers", default=3, help='Number of cross layers for DCNv2')
    flags.DEFINE_integer("cross_layer_projection_dim", default=512, help='Projection dimension used in the cross layers')


define_dcnv2_specific_flags()
import main

def _main(argv):
    main.main()

if __name__ == '__main__':
    app.run(_main)
