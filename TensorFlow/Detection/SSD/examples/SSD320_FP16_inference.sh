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

PIPELINE_CONFIG_PATH=${1:-"/workdir/models/research/configs"}"/ssd320_full_1gpus.config"

export TF_ENABLE_AUTO_MIXED_PRECISION=1

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
OBJECT_DETECTION=$(realpath $SCRIPT_DIR/../object_detection/)
PYTHONPATH=$PYTHONPATH:$OBJECT_DETECTION

python $SCRIPT_DIR/SSD320_inference.py \
       --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
       "${@:2}"
