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


# Copyright 2020 Chengxi Zang
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


from logging import getLogger
import traceback
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from moflow.data.encoding import MolEncoder, EncodingError
from moflow.data.data_loader import NumpyTupleDataset


class DataFrameParser:
    """
    This DataFrameParser parses pandas dataframe containing SMILES and, optionally, some additional features.

    Args:
        encoder (MolEncoder): encoder instance
        labels (list): labels column that should be loaded
        smiles_col (str): smiles column
    """

    def __init__(self, encoder: MolEncoder,
                 labels: List[str],
                 smiles_col: str = 'smiles'):
        super(DataFrameParser, self).__init__()
        self.labels = labels
        self.smiles_col = smiles_col
        self.logger = getLogger(__name__)
        self.encoder = encoder

    def parse(self, df: pd.DataFrame) -> NumpyTupleDataset:
        """Parse DataFrame using `encoder` and prepare a dataset instance

        Labels are extracted from `labels` columns and input features are
        extracted from smiles information in `smiles` column.
        """        
        all_nodes = []
        all_edges = []

        total_count = df.shape[0]
        fail_count = 0
        success_count = 0
        for smiles in tqdm(df[self.smiles_col], total=df.shape[0]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    fail_count += 1
                    continue
                # Note that smiles expression is not unique.
                # we obtain canonical smiles
                nodes, edges = self.encoder.encode_mol(mol)

            except EncodingError as e:
                fail_count += 1
                continue
            except Exception as e:
                self.logger.warning('parse(), type: {}, {}'
                                .format(type(e).__name__, e.args))
                self.logger.info(traceback.format_exc())
                fail_count += 1
                continue
            all_nodes.append(nodes)
            all_edges.append(edges)
            success_count += 1
        
        result = [np.array(all_nodes), np.array(all_edges), *(df[label_col].values for label_col in self.labels)]
        self.logger.info('Preprocess finished. FAIL {}, SUCCESS {}, TOTAL {}'
                    .format(fail_count, success_count, total_count))
  
        dataset = NumpyTupleDataset(result)
        return dataset
