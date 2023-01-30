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

from typing import Any, Dict, List, Tuple, Union

import pandas as pd

try:
    import ipywidgets as widgets
    from IPython import get_ipython
    from IPython.core.display import HTML, Markdown, display
except ImportError:
    print("IPython not installed.")
from typing import Dict


def load_data(
    path_real: str,
    path_fake: str,
    real_sep: str = ",",
    fake_sep: str = ",",
    drop_columns: List = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from a real and synthetic data csv. This function makes sure that the loaded data has the same columns
    with the same data types.
    Args:
        path_real: string path to csv with real data
        path_fake: string path to csv with real data
        real_sep: separator of the real csv
        fake_sep: separator of the fake csv
        drop_columns: names of columns to drop.
    Return: Tuple with DataFrame containing the real data and DataFrame containing the synthetic data.
    """
    real = pd.read_csv(path_real, sep=real_sep, low_memory=False)
    fake = pd.read_csv(path_fake, sep=fake_sep, low_memory=False)
    if set(fake.columns.tolist()).issubset(set(real.columns.tolist())):
        real = real[fake.columns]
    elif drop_columns is not None:
        real = real.drop(drop_columns, axis=1)
        try:
            fake = fake.drop(drop_columns, axis=1)
        except:
            print(f"Some of {drop_columns} were not found on fake.index.")
        assert len(fake.columns.tolist()) == len(
            real.columns.tolist()
        ), f"Real and fake do not have same nr of columns: {len(fake.columns)} and {len(real.columns)}"
        fake.columns = real.columns
    else:
        fake.columns = real.columns

    for col in fake.columns:
        fake[col] = fake[col].astype(real[col].dtype)
    return real, fake


def dict_to_df(data: Dict[str, Any]):
    return pd.DataFrame(
        {"result": list(data.values())}, index=list(data.keys())
    )


class EvaluationResult(object):
    def __init__(
        self, name, content, prefix=None, appendix=None, notebook=False
    ):
        self.name = name
        self.prefix = prefix
        self.content = content
        self.appendix = appendix
        self.notebook = notebook

    def show(self):
        if self.notebook:
            output = widgets.Output()
            with output:
                display(Markdown(f"## {self.name}"))
                if self.prefix:
                    display(Markdown(self.prefix))
                display(self.content)
                if self.appendix:
                    display(Markdown(self.appendix))
            return output

        else:
            print(f"\n{self.name}")
            if self.prefix:
                print(self.prefix)
            print(self.content)
            if self.appendix:
                print(self.appendix)
