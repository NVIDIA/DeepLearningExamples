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
import shutil
import urllib.request
from typing import Any, Callable
from zipfile import ZipFile

from retrying import retry
from tqdm.auto import tqdm

# method from PEP-366 to support relative import in executed modules
if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from .logger import LOGGER
from .exceptions import RunnerException


def unzip(checkpoint_path: pathlib.Path, archive_path: pathlib.Path) -> None:
    """
    Unzip acrhive to provided path

    Args:
        checkpoint_path: Path where archive has to be unpacked
        archive_path: Path to archive Archive filename

    Returns:
        None
    """
    LOGGER.info(f"Creating directory for checkpoint: {checkpoint_path.name}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Unpacking checkpoint files {checkpoint_path}")
    with ZipFile(archive_path, "r") as zf:
        zf.extractall(path=checkpoint_path)
    LOGGER.info("done")

    LOGGER.info(f"Removing zip file: {archive_path}")
    archive_path.unlink()
    LOGGER.info("done")


def download_progress(t: Any) -> Callable:
    """
    Progress bar

    Args:
        t: progress

    Returns:
        Callable
    """
    last_b = [0]

    def update_to(b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize not in (None, -1):
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


@retry(stop_max_attempt_number=3)
def download(checkpoint_url: str, checkpoint_path: pathlib.Path) -> None:
    """
    Download checkpoint from given url to provided path
    Args:
        checkpoint_url: Url from which checkpoint has to be downloaded
        checkpoint_path: Path where checkpoint has to be stored

    Returns:
        None
    """
    LOGGER.info(f"Downloading checkpoint from {checkpoint_url}")
    with tqdm(unit="B") as t:
        reporthook = download_progress(t)
        result = urllib.request.urlretrieve(checkpoint_url, reporthook=reporthook)

    filename = result[0]
    LOGGER.info(f"Checkpoint saved in {filename}")

    file_path = pathlib.Path(filename)
    if not file_path.is_file() and not file_path.is_dir():
        raise RunnerException(f"Checkpoint {filename} does not exist")

    LOGGER.info(f"Moving checkpoint to {checkpoint_path.parent}")
    shutil.move(file_path, checkpoint_path.parent / file_path.name)
    LOGGER.info("done")

    archive_path = checkpoint_path.parent / file_path.name
    unzip(checkpoint_path, archive_path)
