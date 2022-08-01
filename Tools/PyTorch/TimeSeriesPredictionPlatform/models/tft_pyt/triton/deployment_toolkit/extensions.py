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

import importlib
import logging
import os
import re
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)


class ExtensionManager:
    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register_extension(self, extension: str, clazz):
        already_registered_class = self._registry.get(extension, None)
        if already_registered_class and already_registered_class.__module__ != clazz.__module__:
            raise RuntimeError(
                f"Conflicting extension {self._name}/{extension}; "
                f"{already_registered_class.__module__}.{already_registered_class.__name} "
                f"and "
                f"{clazz.__module__}.{clazz.__name__}"
            )
        elif already_registered_class is None:
            clazz_full_name = f"{clazz.__module__}.{clazz.__name__}" if clazz is not None else "None"
            LOGGER.debug(f"Registering extension {self._name}/{extension}: {clazz_full_name}")
            self._registry[extension] = clazz

    def get(self, extension):
        if extension not in self._registry:
            raise RuntimeError(f"Missing extension {self._name}/{extension}")
        return self._registry[extension]

    @property
    def supported_extensions(self):
        return list(self._registry)

    @staticmethod
    def scan_for_extensions(extension_dirs: List[Path]):
        register_pattern = r".*\.register_extension\(.*"

        for extension_dir in extension_dirs:
            for python_path in extension_dir.rglob("*.py"):
                if not python_path.is_file():
                    continue
                payload = python_path.read_text()
                if re.findall(register_pattern, payload):
                    import_path = python_path.relative_to(toolkit_root_dir.parent)
                    package = import_path.parent.as_posix().replace(os.sep, ".")
                    package_with_module = f"{package}.{import_path.stem}"
                    spec = importlib.util.spec_from_file_location(name=package_with_module, location=python_path)
                    my_module = importlib.util.module_from_spec(spec)
                    my_module.__package__ = package

                    try:
                        spec.loader.exec_module(my_module)  # pytype: disable=attribute-error
                    except ModuleNotFoundError as e:
                        LOGGER.error(
                            f"Could not load extensions from {import_path} due to missing python packages; {e}"
                        )


runners = ExtensionManager("runners")
loaders = ExtensionManager("loaders")
savers = ExtensionManager("savers")
converters = ExtensionManager("converters")
toolkit_root_dir = (Path(__file__).parent / "..").resolve()
ExtensionManager.scan_for_extensions([toolkit_root_dir])
