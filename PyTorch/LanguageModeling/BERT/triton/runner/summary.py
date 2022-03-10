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

import csv
import json
import pathlib
from typing import Dict, List, Union

# method from PEP-366 to support relative import in executed modules
import yaml

if __name__ == "__main__" and __package__ is None:
    __package__ = pathlib.Path(__file__).parent.name

from ..deployment_toolkit.report import save_results, sort_results
from .logger import LOGGER


def save_summary(result_type: str, results: List, summary_dir: pathlib.Path) -> None:
    """
    Create file with summary for results of given type
    Args:
        result_type: Type of results to dump
        results: Results data
        summary_dir: Path where results should be stored

    Returns:
        None
    """
    if len(results) == 0:
        LOGGER.warning(f"No {result_type} results found.")
        return

    results = sort_results(results=results)

    kind_file = summary_dir / f"{result_type}_summary.csv"
    save_results(filename=kind_file.as_posix(), data=results, formatted=True)
    LOGGER.info(f"Summary for {result_type} stored in {kind_file}")


def load_results(*, results_path: Union[pathlib.Path, str], result_type: str, parameters: Dict) -> List:
    """
    Update results
    Args:
        results_path: Path to file or directory from which data should be read
        result_type: type of results
        parameters: Parameters used in experiment which generated results


    Returns:
        List of result rows
    """
    LOGGER.debug(f"Loading {result_type} from {results_path} for summary")
    results_path = pathlib.Path(results_path)

    if results_path.is_file():
        files = [results_path]
    elif results_path.is_dir():
        files = list(results_path.iterdir())
    else:
        LOGGER.debug(f"Unable to load file: {results_path}. Generating empty rows.")
        data = [{}]
        return data

    if any([file.name.endswith(".ckpt") for file in files]):
        model_analyzer_metrics = results_path / "metrics-model-inference.csv"
        files = [model_analyzer_metrics]
    else:
        files = [file for file in files if file.name.endswith(".csv")]

    results = list()
    parameters_cpy = {key: value for key, value in parameters.items() if key != "batch"}
    for file in files:
        if file.suffix == ".csv":
            data = _generate_data_from_csv(file=file)
        elif file.suffix == ".json":
            data = _generate_data_from_json(file=file)
        elif file.suffix == ".yaml":
            data = _generate_data_from_yaml(file=file)
        else:
            raise ValueError(f"Unsupported file extension: {file.suffix}")

        for item in data:
            result = {**parameters_cpy, **item}
            results.append(result)

    LOGGER.debug(f"Loading done. Collected {len(results)} results.")
    return results


def _normalize_key(*, key: str) -> str:
    """
    Normalize key

    Args:
        key: Key to normalize

    Returns:
        Normalized string
    """
    key = "_".join(key.split(sep=" "))
    key = key.lower()
    return key


def _normalize_keys(*, data: Dict) -> Dict:
    """
    Normalize keys in dictionary

    Args:
        data: Dictionary to normalize

    Returns:
        Normalized dictionary
    """
    keys = {_normalize_key(key=key): value for key, value in data.items()}
    return keys


def _generate_data_from_csv(*, file: Union[pathlib.Path, str]) -> List[Dict]:
    """
    Generate result rows from CSV file
    Args:
        file: CSV file path

    Returns:
        List of rows
    """
    LOGGER.debug(f"Reading data from {file}")
    filtered_rows: List[Dict] = []
    with open(file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            r = _normalize_keys(data=r)
            filtered_row = {k: v for k, v in r.items()}
            filtered_rows.append(filtered_row)

    LOGGER.debug("done")

    return filtered_rows


def _generate_data_from_json(file: pathlib.Path) -> List[Dict]:
    LOGGER.info(f"Reading data from {file}")
    filtered_rows: List[Dict] = list()
    with open(file, "r") as json_file:
        file_data = json.load(json_file)
        if not isinstance(file_data, list):
            file_data = [file_data]

    for r in file_data:
        r = _normalize_keys(data=r)
        filtered_row = {k: v for k, v in r.items()}
        filtered_rows.append(filtered_row)

    LOGGER.info("done")

    return filtered_rows


def _generate_data_from_yaml(file: pathlib.Path) -> List[Dict]:
    LOGGER.info(f"Reading data from {file}")
    filtered_rows: List[Dict] = list()
    with open(file, "r") as yaml_file:
        file_data = yaml.safe_load(yaml_file)
        if not isinstance(file_data, list):
            file_data = [file_data]

    for r in file_data:
        r = _normalize_keys(data=r)
        filtered_row = {k: v for k, v in r.items()}
        filtered_rows.append(filtered_row)

    LOGGER.info("done")

    return filtered_rows
