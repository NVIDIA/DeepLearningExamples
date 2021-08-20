from setuptools import setup, find_packages

setup(
    name='se3-transformer',
    packages=find_packages(),
    include_package_data=True,
    version='1.0.0',
    description='PyTorch + DGL implementation of SE(3)-Transformers',
    author='Alexandre Milesi',
    author_email='alexandrem@nvidia.com',
)
