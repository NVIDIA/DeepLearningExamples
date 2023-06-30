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

import json
import os
import tensorflow as tf
from tensorflow.python.saved_model import save_options


from nn.embedding import DualEmbeddingGroup

class Model(tf.keras.Model):
    def __init__(self, cardinalities, output_dim, memory_threshold):
        super().__init__()
        self.cardinalities = cardinalities
        self.output_dim = output_dim
        self.embedding = DualEmbeddingGroup(cardinalities, output_dim, memory_threshold, use_mde_embeddings=False)

    @tf.function
    def call(self, x):
        x = self.embedding(x)
        x = tf.reshape(x, [-1, len(self.cardinalities) * self.output_dim])
        return x

_sparse_model_config_template = r"""name: "{model_name}"
platform: "tensorflow_savedmodel"
max_batch_size:{max_batch_size}
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator {{
      name: "gpu_io"
    }}
  }}
}}
version_policy: {{
        specific:{{versions: {version}}}
}},
instance_group [
  {{
    count: {engine_count_per_device}
    kind : KIND_GPU
    gpus : [0]
  }}
]"""


def save_triton_config(
    dst_path, model_name, version, max_batch_size, engine_count_per_device
):
    config_str = _sparse_model_config_template.format(
        model_name=model_name,
        max_batch_size=max_batch_size,
        version=version,
        engine_count_per_device=engine_count_per_device,
    )

    with open(dst_path, "w") as f:
        f.write(config_str)
    print("Wrote sparse model Triton config to:", dst_path)


def deploy_sparse(
    src,
    dst,
    model_name,
    max_batch_size,
    engine_count_per_device,
    memory_threshold_gb,
    num_gpus=1,
    version="1",
    **kwargs,
):
    print("deploy sparse dst: ", dst)
    with open(os.path.join(src, "config.json")) as f:
        src_config = json.load(f)

    model = Model(cardinalities=src_config["categorical_cardinalities"],
                  output_dim=src_config['embedding_dim'][0],
                  memory_threshold=memory_threshold_gb)

    x = tf.zeros(shape=(65536, len(src_config["categorical_cardinalities"])), dtype=tf.int32)
    _ = model(x)

    model.embedding.restore_checkpoint(src)

    options = save_options.SaveOptions(experimental_variable_policy=save_options.VariablePolicy.SAVE_VARIABLE_DEVICES)
    savedmodel_dir = os.path.join(dst, '1', 'model.savedmodel')
    os.makedirs(savedmodel_dir)
    tf.keras.models.save_model(model=model, filepath=savedmodel_dir, overwrite=True, options=options)

    save_triton_config(
        dst_path=os.path.join(dst, "config.pbtxt"),
        model_name=model_name,
        version=version,
        max_batch_size=max_batch_size,
        engine_count_per_device=engine_count_per_device,
    )

    return len(src_config["categorical_cardinalities"])
