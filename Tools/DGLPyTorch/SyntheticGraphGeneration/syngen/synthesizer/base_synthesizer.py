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


class BaseSynthesizer(abc.ABC):
    """Base class for all ``Synthesizers``"""

    @classmethod
    def get_synthesizers(cls, include_parents=True):
        """Recursively find sublcasses of `BaseSynthesizer`
        
        Args:
            include_parents (bool): whether to include parents to other classes. (default: `True`)
        """

        synthesizers = dict()
        for child in cls.__subclasses__():
            children = child.get_synthesizers(include_parents)
            synthesizers.update(children)

            if include_parents or not children:
                if abc.ABC not in child.__bases__:
                    synthesizers[child.__name__] = child
        return synthesizers

    def fit(self, *args, **kwargs):
        """fits synthesizer on a specified dataset"""
        raise NotImplementedError()

    def generate(self, *args, **kwargs):
        """generate graph using configured synthesizer"""
        raise NotImplementedError()

    def save(self, path: str):
        """save this synthesizer to disk
        Args:
            path (str): The path to save the synthesizer to
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str):
        """load up a saved synthesizer object from disk.

        Args:
            path (str): The path to load the synthesizer from
        """
        raise NotImplementedError()

    @staticmethod
    def add_args(parser):
        """optional function to add arguments to parser for the CLI interface"""
        return parser
