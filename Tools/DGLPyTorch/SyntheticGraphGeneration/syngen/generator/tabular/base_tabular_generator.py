# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import abc

import torch


class BaseTabularGenerator(abc.ABC):
    """Base class for all tabular generators"""

    def __init__(self, **kwargs):
        pass

    @classmethod
    def get_generators(cls, include_parents=True):
        """Recursively find subclasses of `BaseTabularGenerator`
        
        Args:
            include_parents (bool): whether to include parents to other classes. (default: `True`)
        """

        generators = dict()
        for child in cls.__subclasses__():
            children = child.get_generators(include_parents)
            generators.update(children)

            if include_parents or not children:
                if abc.ABC not in child.__bases__:
                    generators[child.__name__] = child
        return generators

    def fit(self, *args, **kwargs):
        """fit function for the generator

        Args:
            *args: optional positional args
            **kwargs: optional key-word arguments
        """
        raise NotImplementedError()

    def sample(self, num_samples, *args, **kwargs):
        """generate `num_samples` from generator

        Args:
            num_samples (int): number of samples to generate
            *args: optional positional args
            **kwargs: optional key-word arguments
        """
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    @property
    def supports_memmap(self) -> bool:
        return False

    @classmethod
    def load(cls, path):
        raise NotImplementedError()

    @staticmethod
    def add_args(parser):
        return parser
