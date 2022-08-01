#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#
#-------------------------------------------------------------------------
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()


extra_compile_args = {'cxx' : ['-O2']}
extra_compile_args['nvcc'] = ['-O3',
                              '-I./cutlass/',
                              '-U__CUDA_NO_HALF_OPERATORS__',
                              '-U__CUDA_NO_HALF_CONVERSIONS__',
                              '-gencode', 'arch=compute_70,code=sm_70',
                              '-gencode', 'arch=compute_70,code=compute_70',
                              '-gencode', 'arch=compute_80,code=sm_80',
                              '-gencode', 'arch=compute_80,code=compute_80',
                              ]

strided_batched_gemm = CUDAExtension(
                        name='strided_batched_gemm',
                        sources=['fairseq/modules/strided_batched_gemm/strided_batched_gemm.cpp', 'fairseq/modules/strided_batched_gemm/strided_batched_gemm_cuda.cu'],
                        extra_compile_args=extra_compile_args
)

batch_utils = CppExtension(
                        name='fairseq.data.batch_C',
                        sources=['fairseq/data/csrc/make_batches.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                        }
)
setup(
    name='fairseq',
    version='0.5.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    long_description=readme,
    license=license,
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[strided_batched_gemm, batch_utils],
    cmdclass={
                'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    test_suite='tests',
)
