# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
class ModelAnalyzerException(Exception):
    def __init__(self, message: str):
        self._message = message

    def __str__(self):
        """
        Get the exception string representation.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.
        """
        return self._message

    @property
    def message(self):
        """
        Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.
        """
        return self._message
