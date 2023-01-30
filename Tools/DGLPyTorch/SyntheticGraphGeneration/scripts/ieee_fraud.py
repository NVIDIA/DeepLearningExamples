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

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    data_path = sys.argv[1]
    # - path containing ieee-fraud-detection data
    # https://www.kaggle.com/competitions/ieee-fraud-detection
    data_path = Path(data_path)

    # - concat data files

    train_trn = pd.read_csv(data_path / 'train_transaction.csv')
    test_trn = pd.read_csv(data_path / 'test_transaction.csv')
    # - not every transactionID has an associated transaction identification ...
    data = pd.concat([train_trn, test_trn], axis=0)

    user_cols = ['addr1', 'addr2', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6']
    # - product columns that can be used to create unique id
    product_cols = ['ProductCD', 'R_emaildomain']
    for c in user_cols:
        data.loc[:, c] = data[c].fillna('').astype(str)

    for c in product_cols:
        data.loc[:, c] = data[c].fillna('').astype(str)

    data['user_id'] = ''
    user_cols_selected = ['card1'] # - select only card1
    for c in user_cols_selected:
        data.loc[:, 'user_id'] = data['user_id'] + data[c]

    data['product_id'] = ''
    for c in product_cols:
        data.loc[:, 'product_id'] = data['product_id'] + data[c]

    # - drop id cols
    data.drop(columns=user_cols + product_cols, inplace=True)

    # - select last transaction
    data = data.sort_values('TransactionDT').groupby(['user_id', 'product_id']).tail(1)

    # - dump data
    save_path = os.path.join(data_path, 'data.csv')
    data.to_csv(save_path, index=False)
