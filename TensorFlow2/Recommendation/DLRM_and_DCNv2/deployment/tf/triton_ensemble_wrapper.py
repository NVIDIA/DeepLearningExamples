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

import tritonclient.utils
import tritonclient.http
import numpy as np
import tensorflow as tf

import deployment.tf.constants as c


class RecsysTritonEnsemble:
    def __init__(self, model_name, num_tables, verbose, categorical_sizes, fused_embedding=True):
        self.model_name = model_name
        self.triton_client = tritonclient.http.InferenceServerClient(url="localhost:8000", verbose=verbose)
        if not self.triton_client.is_server_live():
            raise ValueError('Triton server is not live!')

        print('triton model repo: ', self.triton_client.get_model_repository_index())

    def __call__(self, inputs, sigmoid=False, training=False):
        numerical_features, cat_features = list(inputs.values())

        batch_size = cat_features[0].shape[0]

        cat_features = tf.concat(cat_features, axis=1).numpy().astype(np.int32)
        numerical_features = numerical_features.numpy().astype(np.float32)

        inputs = [
            tritonclient.http.InferInput("categorical_features",
                                  cat_features.shape,
                                  tritonclient.utils.np_to_triton_dtype(np.int32)),
            tritonclient.http.InferInput("numerical_features",
                                  numerical_features.shape,
                                  tritonclient.utils.np_to_triton_dtype(np.float32)),
        ]
        inputs[0].set_data_from_numpy(cat_features)
        inputs[1].set_data_from_numpy(numerical_features)

        outputs = [tritonclient.http.InferRequestedOutput(c.ens_output_name)]

        response = self.triton_client.infer(self.model_name, inputs, outputs=outputs)

        result_np = response.as_numpy(c.ens_output_name)
        result_np = result_np.reshape([batch_size])
        return result_np
