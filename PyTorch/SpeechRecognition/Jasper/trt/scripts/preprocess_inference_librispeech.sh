#!/usr/bin/env bash
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

# Constructs JSON manifests for inference-subset of Librispeech corpus.

python ./utils/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/dev-clean \
    --dest_dir /datasets/LibriSpeech/dev-clean-wav \
    --output_json /datasets/LibriSpeech/librispeech-dev-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/dev-other \
    --dest_dir /datasets/LibriSpeech/dev-other-wav \
    --output_json /datasets/LibriSpeech/librispeech-dev-other-wav.json


python ./utils/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/test-clean \
    --dest_dir /datasets/LibriSpeech/test-clean-wav \
    --output_json /datasets/LibriSpeech/librispeech-test-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir /datasets/LibriSpeech/test-other \
    --dest_dir /datasets/LibriSpeech/test-other-wav \
    --output_json /datasets/LibriSpeech/librispeech-test-other-wav.json
