# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

abspath = os.path.dirname(os.path.realpath(__file__))

print(find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))

setup(name="dlrm",
      package_dir={'dlrm': 'dlrm'},
      version="1.0.0",
      description="Reimplementation of Facebook's DLRM",
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      zip_safe=False,
      ext_modules=[
          CUDAExtension(name="dlrm.cuda_ext.fused_embedding",
                        sources=[
                            os.path.join(abspath, "dlrm/cuda_src/pytorch_embedding_ops.cpp"),
                            os.path.join(abspath, "dlrm/cuda_src/gather_gpu_fused_pytorch_impl.cu")
                        ],
                        extra_compile_args={
                            'cxx': [],
                            'nvcc': ["-arch=sm_70",
                                     '-gencode', 'arch=compute_80,code=sm_80']
                        }),
          CUDAExtension(name="dlrm.cuda_ext.interaction_volta",
                        sources=[
                            os.path.join(abspath, "dlrm/cuda_src/dot_based_interact_volta/pytorch_ops.cpp"),
                            os.path.join(abspath, "dlrm/cuda_src/dot_based_interact_volta/dot_based_interact_pytorch_types.cu")
                        ],
                        extra_compile_args={
                            'cxx': [],
                            'nvcc': [
                                '-DCUDA_HAS_FP16=1',
                                '-D__CUDA_NO_HALF_OPERATORS__',
                                '-D__CUDA_NO_HALF_CONVERSIONS__',
                                '-D__CUDA_NO_HALF2_OPERATORS__',
                                '-gencode', 'arch=compute_70,code=sm_70']
                        }),
          CUDAExtension(name="dlrm.cuda_ext.interaction_ampere",
                        sources=[
                            os.path.join(abspath, "dlrm/cuda_src/dot_based_interact_ampere/pytorch_ops.cpp"),
                            os.path.join(abspath, "dlrm/cuda_src/dot_based_interact_ampere/dot_based_interact_pytorch_types.cu")
                        ],
                        extra_compile_args={
                            'cxx': [],
                            'nvcc': [
                                '-DCUDA_HAS_FP16=1',
                                '-D__CUDA_NO_HALF_OPERATORS__',
                                '-D__CUDA_NO_HALF_CONVERSIONS__',
                                '-D__CUDA_NO_HALF2_OPERATORS__',
                                '-gencode', 'arch=compute_80,code=sm_80']
                        }),
          CUDAExtension(name="dlrm.cuda_ext.sparse_gather",
                        sources=[
                            os.path.join(abspath, "dlrm/cuda_src/sparse_gather/sparse_pytorch_ops.cpp"),
                            os.path.join(abspath, "dlrm/cuda_src/sparse_gather/gather_gpu.cu")
                        ],
                        extra_compile_args={
                            'cxx': [],
                            'nvcc': ["-arch=sm_70",
                                     '-gencode', 'arch=compute_80,code=sm_80']
                        })
      ],
      cmdclass={"build_ext": BuildExtension})
