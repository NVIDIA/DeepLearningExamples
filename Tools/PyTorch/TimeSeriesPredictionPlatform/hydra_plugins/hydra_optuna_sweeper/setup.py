# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="hydra-optuna-sweeper",
    version=read_version("hydra_plugins/hydra_optuna_sweeper", "__init__.py"),
    author="Toshihiko Yanase, Hiroyuki Vincent Yamazaki",
    author_email="toshihiko.yanase@gmail.com, hiroyuki.vincent.yamazaki@gmail.com",
    description="Hydra Optuna Sweeper plugin",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/hydra/",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "hydra-core>=1.1.0.dev7",
        "optuna>=3.0.0",
    ],
    include_package_data=True,
)
