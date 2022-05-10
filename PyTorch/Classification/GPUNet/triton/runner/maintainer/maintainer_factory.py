# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import pathlib

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .docker.maintainer import DockerMaintainer


class MaintainerFactory:
    @staticmethod
    def create_docker_maintainer():
        return DockerMaintainer()
