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
import math
import os
import numpy as np


def save_embedding_table(numpy_table, dst_dir, offset=0, min_keys=100):
    if numpy_table.shape[0] < min_keys:
        print(
            f"Artificially lengthening embedding table from size: {numpy_table.shape} to size {min_keys}"
        )
        num_missing_rows = min_keys - numpy_table.shape[0]
        padding = np.zeros(
            shape=[num_missing_rows, numpy_table.shape[1]], dtype=numpy_table.dtype
        )
        numpy_table = np.vstack([numpy_table, padding])

    keys_table = np.arange(
        start=offset, stop=offset + numpy_table.shape[0], dtype=np.int64
    )
    keys_bytes = keys_table.tobytes()
    key_file = os.path.join(dst_dir, "key")
    with open(key_file, "wb") as f:
        f.write(keys_bytes)

    table_bytes = numpy_table.tobytes()
    table_file = os.path.join(dst_dir, "emb_vector")
    with open(table_file, "wb") as f:
        f.write(table_bytes)


_hps_triton_config_template = r"""name: "{model_name}"
backend: "hps"
max_batch_size:{max_batch_size}
input [  {{
    name: "KEYS"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "NUMKEYS"
    data_type: TYPE_INT32
    dims: [-1]
  }}]
output [  {{
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [-1]
  }}]
version_policy: {{
        specific:{{versions: {version}}}
}},
instance_group [
  {{
    count: {engine_count_per_device}
    kind : KIND_GPU
    gpus : [0]
  }}
]
"""


def save_triton_config(
    dst_path, model_name, version, max_batch_size, engine_count_per_device
):
    config = _hps_triton_config_template.format(
        model_name=model_name,
        max_batch_size=max_batch_size,
        version=version,
        engine_count_per_device=engine_count_per_device,
    )

    print("saving pbtxt HPS config to: ", dst_path)
    with open(dst_path, "w") as f:
        f.write(config)
    print("Wrote HPS Triton config to:", dst_path)

    print(f"{model_name} configuration:")
    print(config)


def save_json_config(
    dst_path,
    hps_embedding_dirs,
    src_config,
    num_gpus,
    gpucacheper,
    max_batch_size,
    model_name,
    fused=True,
):
    num_cat_features = 1 if fused else len(src_config["categorical_cardinalities"])

    if len(hps_embedding_dirs) != num_cat_features:
        raise ValueError(
            f"Length mismatch between hps_embedding_dirs ({len(hps_embedding_dirs)}) "
            f"and num_cat_features ({num_cat_features}), fused={fused}. This should not happen."
        )

    vecsize_per_table = src_config["embedding_dim"]
    max_batch_size_factor = 1
    if fused:
        vecsize_per_table = [vecsize_per_table[0]]
        max_batch_size_factor = len(src_config["categorical_cardinalities"])

    hps_embedding_config = {
        "supportlonglong": True,
        "models": [
            {
                "model": model_name,
                # these directories should contain the "emb_vector" and "keys" files, need to copy them over from the previous location
                "sparse_files": hps_embedding_dirs,
                "num_of_worker_buffer_in_pool": 3,
                "embedding_table_names": [
                    f"sparse_embedding{i}" for i in range(num_cat_features)
                ],
                "embedding_vecsize_per_table": vecsize_per_table,
                # for now, every table uses the same embedding dim
                "maxnum_catfeature_query_per_table_per_sample": [
                    1 for _ in range(num_cat_features)
                ],
                "default_value_for_each_table": [1.0 for _ in range(num_cat_features)],
                "deployed_device_list": list(range(num_gpus)),
                "max_batch_size": max_batch_size * max_batch_size_factor,
                "cache_refresh_percentage_per_iteration": 0.0,
                "hit_rate_threshold": 1.0,
                "gpucacheper": gpucacheper,
                "gpucache": True,
            }
        ],
    }
    print("saving json config to: ", dst_path)
    with open(dst_path, "w") as f:
        json.dump(obj=hps_embedding_config, fp=f, indent=4)


def convert_embedding_tables(src_paths, dst, fused):
    if fused:
        return convert_embedding_tables_fused(src_paths, dst)
    else:
        return convert_embedding_tables_unfused(src_paths, dst)


def convert_embedding_tables_unfused(src_paths, dst):
    hps_embedding_dirs = []
    for src_path in src_paths:
        table_index = int(src_path.split("_")[-1].split(".")[0])
        dst_dir = os.path.join(dst, str(table_index))
        print(f"Converting embedding table: {src_path} to {dst_dir}")

        print(f"Loading source from {src_path}")
        data = np.load(src_path, mmap_mode="r")
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Saving embedding table to {dst_dir}")
        save_embedding_table(numpy_table=data, dst_dir=dst_dir)
        hps_embedding_dirs.append(dst_dir)
    return hps_embedding_dirs


def convert_embedding_tables_fused(src_paths, dst):
    dst_dir = os.path.join(dst, "0")
    os.makedirs(dst_dir, exist_ok=True)

    current_offset = 0
    first_width = None

    key_file = os.path.join(dst_dir, "key")
    table_file = os.path.join(dst_dir, "emb_vector")
    with open(key_file, "wb") as keys_f, open(table_file, "wb") as table_f:
        for src_path in src_paths:
            print(f"Converting table {src_path}")
            data = np.load(src_path, mmap_mode="r")

            if first_width is not None and data.shape[1] != first_width:
                raise ValueError(
                    "Attempting to deploy with a fused embedding but not all embeddings have the same dimension."
                    f"Got embedding dimension: {data.shape[1]}, expected: {first_width}"
                )
            if first_width is None:
                first_width = data.shape[1]

            length = data.shape[0]

            keys_table = np.arange(
                start=current_offset, stop=current_offset + length, dtype=np.int64
            )
            keys_bytes = keys_table.tobytes()
            keys_f.write(keys_bytes)

            # write the table in chunks to minimize memory usage
            chunk_size = 2**20
            num_chunks = math.ceil(length / chunk_size)
            for i in range(num_chunks):
                begin = i * chunk_size
                end = (i + 1) * chunk_size
                end = min(end, length)
                table_bytes = data[begin:end].tobytes()
                table_f.write(table_bytes)

            current_offset += length
    return [dst_dir]


def deploy_sparse(
    src,
    dst,
    model_name,
    max_batch_size,
    engine_count_per_device,
    gpucacheper,
    num_gpus=1,
    version="1",
    fused=True,
    **kwargs
):
    print("deploy sparse dst: ", dst)
    with open(os.path.join(src, "config.json")) as f:
        src_config = json.load(f)

    num_cat_features = len(src_config["categorical_cardinalities"])
    src_paths = [os.path.join(src, f"feature_{i}.npy") for i in range(num_cat_features)]
    hps_embedding_dirs = convert_embedding_tables(
        src_paths=src_paths, dst=os.path.join(dst, version), fused=fused
    )

    save_triton_config(
        dst_path=os.path.join(dst, "config.pbtxt"),
        model_name=model_name,
        version=version,
        max_batch_size=max_batch_size,
        engine_count_per_device=engine_count_per_device,
    )

    save_json_config(
        dst_path=os.path.join(dst, f"{model_name}.json"),
        hps_embedding_dirs=hps_embedding_dirs,
        src_config=src_config,
        num_gpus=num_gpus,
        fused=fused,
        gpucacheper=gpucacheper,
        max_batch_size=max_batch_size,
        model_name=model_name,
    )

    return len(src_config["categorical_cardinalities"])
