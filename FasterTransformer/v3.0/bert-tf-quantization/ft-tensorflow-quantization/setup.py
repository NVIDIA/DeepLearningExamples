################################################################################
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
################################################################################
"""Setup script"""

from setuptools import setup, find_packages

setup(name="TensorFlow_FastTransformer_Quantization",
      package=["ft_tensorflow_quantization"],
      package_dir={'ft_tensorflow_quantization': 'ft_tensorflow_quantization'},
      version="0.1.0",
      description="TensorFlow FasterTransformer Quantization",
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      zip_safe=False)
