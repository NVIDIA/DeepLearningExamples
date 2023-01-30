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

import sys
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    data_path = sys.argv[1]
    save_path = Path(data_path).parent
    save_path = save_path / 'data.csv'
    df = pd.read_csv(data_path)
    df['user'] = df['first'] + df['last']
    df = df.groupby(['user', 'merchant'], axis=0).tail(1).reset_index(drop=True)
    df = df.drop(columns=['user'])
    # - save data
    df.to_csv(save_path, index=False)
