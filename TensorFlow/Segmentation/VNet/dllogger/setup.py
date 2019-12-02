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
import pathlib

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setuptools.setup(
    name="DLLogger",
    version="0.1.0",
    author="NVIDIA Corporation",
    description="NVIDIA DLLogger - logging for Deep Learning applications",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/dllogger",
    packages=["dllogger"],
    install_package_data=True,
    license='Apache2',
    license_file='./LICENSE',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
