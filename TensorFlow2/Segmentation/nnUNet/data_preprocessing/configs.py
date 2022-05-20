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

task = {
    "01": "Task01_BrainTumour",
    "02": "Task02_Heart",
    "03": "Task03_Liver",
    "04": "Task04_Hippocampus",
    "05": "Task05_Prostate",
    "06": "Task06_Lung",
    "07": "Task07_Pancreas",
    "08": "Task08_HepaticVessel",
    "09": "Task09_Spleen",
    "10": "Task10_Colon",
}

patch_size = {
    "01_3d_tf2": [128, 128, 128],
    "02_3d_tf2": [80, 192, 160],
    "03_3d_tf2": [128, 128, 128],
    "04_3d_tf2": [40, 56, 40],
    "05_3d_tf2": [20, 320, 256],
    "06_3d_tf2": [80, 192, 160],
    "07_3d_tf2": [40, 224, 224],
    "08_3d_tf2": [64, 192, 192],
    "09_3d_tf2": [64, 192, 160],
    "10_3d_tf2": [56, 192, 160],
    "01_2d_tf2": [192, 160],
    "02_2d_tf2": [320, 256],
    "03_2d_tf2": [512, 512],
    "04_2d_tf2": [56, 40],
    "05_2d_tf2": [320, 320],
    "06_2d_tf2": [512, 512],
    "07_2d_tf2": [512, 512],
    "08_2d_tf2": [512, 512],
    "09_2d_tf2": [512, 512],
    "10_2d_tf2": [512, 512],
}


spacings = {
    "01_3d_tf2": [1.0, 1.0, 1.0],
    "02_3d_tf2": [1.37, 1.25, 1.25],
    "03_3d_tf2": [1, 0.7676, 0.7676],
    "04_3d_tf2": [1.0, 1.0, 1.0],
    "05_3d_tf2": [3.6, 0.62, 0.62],
    "06_3d_tf2": [1.24, 0.79, 0.79],
    "07_3d_tf2": [2.5, 0.8, 0.8],
    "08_3d_tf2": [1.5, 0.8, 0.8],
    "09_3d_tf2": [1.6, 0.79, 0.79],
    "10_3d_tf2": [3, 0.78, 0.78],
    "11_3d_tf2": [5, 0.741, 0.741],
    "01_2d_tf2": [1.0, 1.0],
    "02_2d_tf2": [1.25, 1.25],
    "03_2d_tf2": [0.7676, 0.7676],
    "04_2d_tf2": [1.0, 1.0],
    "05_2d_tf2": [0.62, 0.62],
    "06_2d_tf2": [0.79, 0.79],
    "07_2d_tf2": [0.8, 0.8],
    "08_2d_tf2": [0.8, 0.8],
    "09_2d_tf2": [0.79, 0.79],
    "10_2d_tf2": [0.78, 0.78],
}

ct_min = {
    "03": -17,
    "06": -1024,
    "07": -96,
    "08": -3,
    "09": -41,
    "10": -30,
    "11": -958,
}

ct_max = {
    "03": 201,
    "06": 325,
    "07": 215,
    "08": 243,
    "09": 176,
    "10": 165.82,
    "11": 93,
}

ct_mean = {"03": 99.4, "06": -158.58, "07": 77.9, "08": 104.37, "09": 99.29, "10": 62.18, "11": -547.7}

ct_std = {"03": 39.36, "06": 324.7, "07": 75.4, "08": 52.62, "09": 39.47, "10": 32.65, "11": 281.08}
