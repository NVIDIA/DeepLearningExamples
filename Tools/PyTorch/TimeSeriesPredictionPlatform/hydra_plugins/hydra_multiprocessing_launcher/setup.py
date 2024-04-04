# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="hydra-multiprocessing-launcher",
    version=read_version("hydra_plugins/hydra_multiprocessing_launcher", "__init__.py"),
    author="Dima Zhylko, Jan Bączek",
    author_email="dzhylko@nvidia.com, jbaczek@nvidia.com",
    description="Multiprocessing Launcher for Hydra apps",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/hydra/",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "hydra-core>=1.1.0.dev7",
        "cloudpickle>=2.0.0",
    ],
    include_package_data=True,
)
