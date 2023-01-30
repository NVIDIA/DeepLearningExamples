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

    @classmethod
    def get_generators(cls, include_parents=True):
        """Recursively find sublcasses of `BaseTabularGenerator`
        
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

    def set_device(self, device):
        """set device for all modules"""
        raise NotImplementedError()

    def save(self, path):
        """save the trained model"""
        device_backup = self._device
        self.set_device(torch.device("cpu"))
        torch.save(self, path)
        self.set_device(device_backup)

    @classmethod
    def load(cls, path):
        """load model from `path`"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(path)
        model.set_device(device)
        return model

    @staticmethod
    def add_args(parser):
        return parser
