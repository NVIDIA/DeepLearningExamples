from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='fused_lamb',
    description="Fused LAMB Optimizer for PyTorch native AMP training",
    packages=find_packages(exclude=('test',)),  # NOQA
    ext_modules=[
        CUDAExtension(
            name='fused_lamb_CUDA',
            sources=[
                'csrc/frontend.cpp',
                'csrc/multi_tensor_l2norm_kernel.cu',
                'csrc/multi_tensor_lamb.cu',
            ],
            extra_compile_args={
                'nvcc': [
                    '-lineinfo',
                    '-O3',
                    '--use_fast_math',
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True),
    },
)
