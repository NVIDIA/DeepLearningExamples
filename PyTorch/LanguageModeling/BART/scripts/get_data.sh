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
# ==============================================================================

data_folder=${1:-"/workspace/bart/data/"}

mkdir -p $data_folder

# Download and unzip the stories directories into data folder from https://cs.nyu.edu/~kcho/DMQA/ for both CNN and Daily Mail.
cd $data_folder && gdown --id 0BwmD_VLjROrfTHk4NFg2SndKcjQ && tar xf cnn_stories.tgz
gdown --id 0BwmD_VLjROrfM1BxdkxVaTY2bWs && tar xf dailymail_stories.tgz

cnn_stories=/workspace/bart/data/cnn/stories
dailymail_stories=/workspace/bart/data/dailymail/stories

cd /workspace/cnn-dailymail && python /workspace/cnn-dailymail/make_datafiles.py $cnn_stories $dailymail_stories && mv cnn_dm ${data_folder}

cd ${data_folder} && wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/xsum.tar.gz && tar -xvf xsum.tar.gz
