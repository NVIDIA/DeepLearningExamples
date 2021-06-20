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
import re
from typing import Dict, List

from natsort import natsorted
from tabulate import tabulate


def sort_results(results: List):
    results = natsorted(results, key=lambda item: [item[key] for key in item.keys()])
    return results


def save_results(filename: str, data: List, formatted: bool = False):
    data = format_data(data=data) if formatted else data
    with open(filename, "a") as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)


def format_data(data: List[Dict]) -> List[Dict]:
    formatted_data = list()
    for item in data:
        formatted_item = format_keys(data=item)
        formatted_data.append(formatted_item)

    return formatted_data


def format_keys(data: Dict) -> Dict:
    keys = {format_key(key=key): value for key, value in data.items()}
    return keys


def format_key(key: str) -> str:
    key = " ".join([k.capitalize() for k in re.split("_| ", key)])
    return key


def show_results(results: List[Dict]):
    headers = list(results[0].keys())
    summary = map(lambda x: list(map(lambda item: item[1], x.items())), results)
    print(tabulate(summary, headers=headers))
