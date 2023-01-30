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


class BaseGraphAligner(abc.ABC):
    """Base class for all graph alignment objects"""

    @classmethod
    def get_aligners(cls, include_parents=True):
        """Recursively find sublcasses of `BaseGraphAligner`
        
        Args:
            include_parents (bool): whether to include parents to other classes. 
            (default: `True`)
        """

        aligners = dict()
        for child in cls.__subclasses__():
            children = child.get_aligners(include_parents)
            aligners.update(children)

            if include_parents or not children:
                if abc.ABC not in child.__bases__:
                    aligners[child.__name__] = child
        return aligners

    def fit(self, *args, **kwargs) -> None:
        """function to fit aligner required to be implemented by aligners"""

        raise NotImplementedError()

    def align(self, *args, **kwargs):
        """align function to align generated graph and generated features, 
        required to be implemented by aligners
        """
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        raise NotImplementedError()

    @staticmethod
    def add_args(parser):
        return parser
