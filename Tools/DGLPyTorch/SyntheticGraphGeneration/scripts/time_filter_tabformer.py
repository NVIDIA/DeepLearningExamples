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

    tabformer_path = sys.argv[1]
    save_path = Path(tabformer_path).parent
    save_path = save_path / 'card_transaction.v2.csv'
    df = pd.read_csv(tabformer_path)
    # - create seconds columns to sort transactions by
    t = df["Time"].str.split(":", expand=True)
    t = t[0].apply(int) * 3600 + t[1].apply(int) * 60
    df.loc[:, "Seconds"] = t
    df['Card ID'] = df["User"].astype(str) + df["Card"].astype(str)
    sorted_df = df.sort_values(by="Seconds")

    # - get last element
    tdf = sorted_df.groupby(by=["Card ID", "Merchant Name"],
                            axis=0).tail(1).reset_index(drop=True)
    tdf = tdf.drop(columns=["Card ID", "Seconds"])

    # - save data
    tdf.to_csv(save_path, index=False)
