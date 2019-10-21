#!/usr/bin/env python

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

import glob
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")

    source_cpu = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))

    print('c++: ', source_cpu)
    print('cuda: ', source_cuda)
    sources = source_cpu
    extension = CppExtension

    define_macros = []

    if CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]
    extra_compile_flags= {'cxx' : []}
    extra_compile_flags['nvcc'] = ['-DCUDA_HAS_FP16=1','-D__CUDA_NO_HALF_OPERATORS__','-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_HALF2_OPERATORS__']

    gencodes = [
                '-gencode', 'arch=compute_52,code=sm_52',
                '-gencode', 'arch=compute_60,code=sm_60',
                '-gencode', 'arch=compute_61,code=sm_61',
                '-gencode', 'arch=compute_70,code=sm_70',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_75,code=compute_75',]

    extra_compile_flags['nvcc'] += gencodes

    ext_modules = [
        extension(
            "SSD._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_flags,
        )
    ]

    return ext_modules


setup(
    name="SSD",
    version="0.1",
    author="slayton",
    url="",
    description="SSD in pytorch",
    packages=find_packages(exclude=("configs", "examples", "test",)),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
