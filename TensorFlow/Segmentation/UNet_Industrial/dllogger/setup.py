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

import setuptools

with open("README.md", "r") as f:
  long_description = f.read()

setuptools.setup(
    name="DLLogger",
    version="0.3.1",
    author="Lukasz Mazurek",
    author_email="lmazurek@nvidia.com",
    description="Tools for logging DL training.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nvlmazurek/DLLogger",
    packages=['dllogger'],
    classifiers=[
      "Programming Language :: Python :: 2",
      "Programming Language :: Python :: 3",
      "License :: BSD",
      "Operating System :: OS Independent",
    ],
    license="BSD",
)
