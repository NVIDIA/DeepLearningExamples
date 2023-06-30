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

import deployment.hps.constants as c


class NumpyToHpsInputConverter:
    def __init__(self, categorical_sizes, fused_embedding=True):
        self.offsets = np.cumsum([0] + categorical_sizes)[:-1]
        self.fused_embedding = fused_embedding

    def __call__(self, numerical_features, cat_features):
        batch_size = cat_features[0].shape[0]

        cat_features = [f.numpy().flatten() for f in cat_features]

        # add the offsets
        if self.fused_embedding:
            cat_features = [f + o for f, o in zip(cat_features, self.offsets)]
        key_tensor = np.concatenate(cat_features, axis=0).astype(np.int64).reshape([1, -1])

        if self.fused_embedding:
            nkey_tensor = np.full(shape=(1, 1), fill_value=batch_size * len(cat_features), dtype=np.int32)
        else:
            nkey_tensor = np.full(shape=(1, len(cat_features)), fill_value=batch_size, dtype=np.int32)

        numerical_features = numerical_features.numpy().astype(np.float32).reshape([1, -1])
        return key_tensor, nkey_tensor, numerical_features


class RecsysTritonEnsemble:
    def __init__(self, model_name, num_tables, verbose, categorical_sizes, fused_embedding=True):
        self.input_converter = NumpyToHpsInputConverter(categorical_sizes, fused_embedding)
        self.model_name = model_name
        self.triton_client = tritonclient.http.InferenceServerClient(url="localhost:8000", verbose=verbose)
        if not self.triton_client.is_server_live():
            raise ValueError('Triton server is not live!')

        print('triton model repo: ', self.triton_client.get_model_repository_index())

    def __call__(self, inputs, sigmoid=False, training=False):
        numerical_features, cat_features = list(inputs.values())
        batch_size = cat_features[0].shape[0]

        key_tensor, nkey_tensor, numerical_features = self.input_converter(numerical_features, cat_features)

        inputs = [
            tritonclient.http.InferInput(c.key_global_prefix,
                                  key_tensor.shape,
                                  tritonclient.utils.np_to_triton_dtype(np.int64)),
            tritonclient.http.InferInput(c.numkey_global_prefix,
                                  nkey_tensor.shape,
                                  tritonclient.utils.np_to_triton_dtype(np.int32)),
            tritonclient.http.InferInput(c.ens_numerical_features_name,
                                         numerical_features.shape,
                                         tritonclient.utils.np_to_triton_dtype(np.float32)),
        ]
        inputs[0].set_data_from_numpy(key_tensor)
        inputs[1].set_data_from_numpy(nkey_tensor)
        inputs[2].set_data_from_numpy(numerical_features)


        outputs = [tritonclient.http.InferRequestedOutput(c.ens_output_name)]
        response = self.triton_client.infer(self.model_name, inputs, outputs=outputs)
        result_np = response.as_numpy(c.ens_output_name)

        result_np = result_np.reshape([batch_size])
        return result_np
