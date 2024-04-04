# Copyright 2021-2024 NVIDIA Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Copyright 2020 The Google Research Authors.
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
"""
Only downloads data if the csv files are present, unless the "force_download"
argument is supplied. For new datasets, the download_and_unzip(.) can be reused
to pull csv files from an online repository, but may require subsequent
dataset-specific processing.

Usage:
  python3 script_download_data --dataset {DATASET} --output_dir {DIR}
Command line args:
  DATASET: Name of dataset to download {e.g. electricity}
  DIR: Path to main dataset diredtory
"""

from __future__ import absolute_import, division, print_function

import argparse
from cmath import nan
import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd
import py7zr
import pyunpack
import wget
import pickle

from datetime import date, timedelta, datetime
from scipy.spatial import distance_matrix

import dgl
import torch

warnings.filterwarnings("ignore")


# General functions for data downloading & aggregation.
def download_from_url(url, output_path):
    """Downloads a file froma url."""

    print("Pulling data from {} to {}".format(url, output_path))
    wget.download(url, output_path)
    print("done")


def unzip(zip_path, output_file, data_folder, use_z=False):
    """Unzips files and checks successful completion."""

    print("Unzipping file: {}".format(zip_path))
    if use_z:
        py7zr.SevenZipFile(zip_path, mode="r").extractall(path=data_folder)
    else:
        pyunpack.Archive(zip_path).extractall(data_folder)

    # Checks if unzip was successful
    if not os.path.exists(output_file):
        raise ValueError(
            "Error in unzipping process! {} not found.".format(output_file)
        )


def download_and_unzip(url, zip_path, csv_path, data_folder):
    """Downloads and unzips an online csv file.

    Args:
      url: Web address
      zip_path: Path to download zip file
      csv_path: Expected path to csv file
      data_folder: Folder in which data is stored.
    """

    download_from_url(url, zip_path)

    unzip(zip_path, csv_path, data_folder)

    print("Done.")


# Dataset specific download routines.
def download_electricity(data_folder):
    """Downloads electricity dataset from UCI repository."""

    url = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"

    csv_path = os.path.join(data_folder, "LD2011_2014.txt")
    zip_path = csv_path + ".zip"

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print("Aggregating to hourly data")

    df = pd.read_csv(csv_path, index_col=0, sep=";", decimal=",")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Used to determine the start and end dates of a series
    output = df.resample("1h").mean().replace(0.0, np.nan)

    earliest_time = output.index.min()
    # Filter to match range used by other academic papers
    output = output[(output.index >= '2014-01-01') & (output.index < '2014-09-08')]

    df_list = []
    for label in output:
        srs = output[label]

        if srs.isna().all():
            continue

        start_date = min(srs.fillna(method="ffill").dropna().index)
        end_date = max(srs.fillna(method="bfill").dropna().index)

        srs = output[label].fillna(0.0)

        tmp = pd.DataFrame({"power_usage": srs})
        date = tmp.index
        tmp["t"] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time
        ).days * 24
        tmp["days_from_start"] = (date - earliest_time).days
        tmp["categorical_id"] = label
        tmp["date"] = date
        tmp["id"] = label
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month
        tmp["power_usage_weight"] = ((date >= start_date) & (date <= end_date))

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join="outer").reset_index(drop=True)

    output["categorical_id"] = output["id"].copy()
    output["hours_from_start"] = output["t"]
    output["categorical_day_of_week"] = output["day_of_week"].copy()
    output["categorical_hour"] = output["hour"].copy()
    output["power_usage_weight"] = output["power_usage_weight"].apply(lambda b: 1 if b else 0)


    output.to_csv(data_folder + "/electricity.csv")

    print("Done.")
    
def download_traffic(data_folder):
    """Downloads traffic dataset from UCI repository."""

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip"

    csv_path = os.path.join(data_folder, "PEMS_train")
    zip_path = os.path.join(data_folder, "PEMS-SF.zip")

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print("Aggregating to hourly data")

    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            parsed_list = [
                variable_type(i)
                for i in s.replace("[", "").replace("]", "").split()
            ]
        else:
            parsed_list = [
                variable_type(i)
                for i in s.replace("[", "").replace("]", "").split(delimiter)
            ]

        return parsed_list

    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), "r") as dat:
            parsed_list_from_file = process_list(dat.readlines()[0])
        return parsed_list_from_file

    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(data_folder, filename), "r") as dat:

            lines = dat.readlines()
            for i, line in enumerate(lines):

                array = [
                    process_list(row_split, variable_type=float, delimiter=None)
                    for row_split in process_list(
                        line, variable_type=str, delimiter=";"
                    )
                ]
                array_list.append(array)

        return array_list

    shuffle_order = np.array(read_single_list("randperm")) - 1  # index from 0
    train_dayofweek = read_single_list("PEMS_trainlabels")
    train_tensor = read_matrix("PEMS_train")
    test_dayofweek = read_single_list("PEMS_testlabels")
    test_tensor = read_matrix("PEMS_test")

    # Inverse permutate shuffle order
    print("Shuffling")
    inverse_mapping = {
        new_location: previous_location
        for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array(
        [
            inverse_mapping[new_location]
            for new_location, _ in enumerate(shuffle_order)
        ]
    )

    # Group and reoder based on permuation matrix
    print("Reodering")
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    # Put everything back into a dataframe
    print("Parsing as dataframe")
    labels = ["traj_{}".format(i) for i in read_single_list("stations_list")]

    hourly_list = []
    for day, day_matrix in enumerate(combined_tensor):

        # Hourly data
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        hourly["hour_on_day"] = [
            int(i / 6) for i in hourly.index
        ]  # sampled at 10 min intervals
        if hourly["hour_on_day"].max() > 23 or hourly["hour_on_day"].min() < 0:
            raise ValueError(
                "Invalid hour! {}-{}".format(
                    hourly["hour_on_day"].min(), hourly["hour_on_day"].max()
                )
            )

        hourly = hourly.groupby("hour_on_day", as_index=True).mean()[labels]
        hourly["sensor_day"] = day
        hourly["time_on_day"] = hourly.index
        hourly["day_of_week"] = day_of_week[day]

        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

    # Flatten such that each entitiy uses one row in dataframe
    store_columns = [c for c in hourly_frame.columns if "traj" in c]
    other_columns = [c for c in hourly_frame.columns if "traj" not in c]
    flat_df = pd.DataFrame(
        columns=["values", "prev_values", "next_values"]
        + other_columns
        + ["id"]
    )

    def format_index_string(x):
        """Returns formatted string for key."""

        if x < 10:
            return "00" + str(x)
        elif x < 100:
            return "0" + str(x)
        elif x < 1000:
            return str(x)

        raise ValueError("Invalid value of x {}".format(x))

    for store in store_columns:

        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ["values"] + other_columns
        sliced["id"] = int(store.replace("traj_", ""))

        # Sort by Sensor-date-time
        key = (
            sliced["id"].apply(str)
            + sliced["sensor_day"].apply(lambda x: "_" + format_index_string(x))
            + sliced["time_on_day"].apply(
                lambda x: "_" + format_index_string(x)
            )
        )
        sliced = sliced.set_index(key).sort_index()

        sliced["values"] = sliced["values"].fillna(method="ffill")
        sliced["prev_values"] = sliced["values"].shift(1)
        sliced["next_values"] = sliced["values"].shift(-1)

        flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)

    # Filter to match range used by other academic papers
    index = flat_df["sensor_day"]
    flat_df = flat_df[index < 173].copy()

    # Creating columns fo categorical inputs
    flat_df["categorical_id"] = flat_df["id"].copy()
    flat_df["hours_from_start"] = (
        flat_df["time_on_day"] + flat_df["sensor_day"] * 24.0
    )
    flat_df["categorical_day_of_week"] = flat_df["day_of_week"].copy()
    flat_df["categorical_time_on_day"] = flat_df["time_on_day"].copy()

    flat_df.to_csv(data_folder + "/traffic.csv")

def download_m5(data_folder):
    """Processes M5 Kaggle competition dataset.

    Raw files can be manually downloaded from Kaggle (without test set) @
      https://www.kaggle.com/c/m5-forecasting-accuracy/data
    
    Data is downloaded from Google Drive from organizers @
      https://github.com/Mcompetitions/M5-methods

    Args:
      config: Default experiment config for M5
    """
    required_files = ['sales_train_evaluation.csv', 'sales_test_evaluation.csv', 
                      'sell_prices.csv', 'calendar.csv', 'weights_validation.csv', 
                      'weights_evaluation.csv']
    
    for file in required_files:
        assert os.path.exists(os.path.join(data_folder, file)), "There are files missing from the data_folder. Please download following files from https://github.com/Mcompetitions/M5-methods"

    core_frame = pd.read_csv(os.path.join(data_folder, "sales_train_evaluation.csv"))
    test_frame = pd.read_csv(os.path.join(data_folder, "sales_test_evaluation.csv"))
    # Add 28 prediction values for final model evaluation
    core_frame = core_frame.merge(test_frame, on=['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    del test_frame

    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ts_cols = [col for col in core_frame.columns if col not in id_vars]

    core_frame['id'] = core_frame.item_id + '_' + core_frame.store_id
    prices = pd.read_csv(os.path.join(data_folder, "sell_prices.csv"))
    calendar = pd.read_csv(os.path.join(data_folder, "calendar.csv"))

    calendar = calendar.sort_values('date')
    calendar['d'] = [f'd_{i}' for i in range(1, calendar.shape[0]+1)]

    core_frame = core_frame.melt(
        id_vars,
        value_vars=ts_cols,
        var_name='d',
        value_name='items_sold'
    )
    core_frame = core_frame.merge(calendar, left_on="d", right_on='d')
    core_frame = core_frame.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='outer')

    # According to M5-Comperition-Guide-Final-10-March-2020:
    # if not available, this means that the product was not sold during the examined week.
    core_frame.sell_price.fillna(-1, inplace=True)

    core_frame['weight'] = 1.0
    core_frame.loc[core_frame.sell_price == -1, 'weight'] = 0

    core_frame.to_csv(os.path.join(data_folder, "M5.csv"))

def construct_graph(nodes_loc, k=0.8):
    """
    Constructs a graph based on a physical location of nodes
    nodes_loc: 2D array num_nodes x dim
    features: list of node features
    """
    dist_mx = distance_matrix(nodes_loc, nodes_loc)

    std = dist_mx.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    adj_mx[adj_mx < k] = 0
    np.fill_diagonal(adj_mx, 0)

    edges = np.nonzero(adj_mx)
    graph = dgl.graph(edges, num_nodes=nodes_loc.shape[0])
    return graph

def download_PEMS_BAY(data_folder):
    def check_completeness(data_folder):
        """Returns list of raw data files"""

        def daterange(start_date, end_date):
            for n in range(int((end_date - start_date).days)):
                yield start_date + timedelta(n)

        start_date = date(2017, 1, 1)
        end_date = date(2017, 7, 1)
        fnames = ['d04_text_station_5min_' + d.strftime('%Y_%m_%d') + '.txt.gz' for d in daterange(start_date, end_date)]
        missing = set(fnames).difference(os.listdir(data_folder))
        assert not missing, f"""There are files missing from the data_folder.
                               Please download following files from https://pems.dot.ca.gov/?dnode=Clearinghouse
                               {missing}"""

        fnames = [os.path.join(data_folder, f) for f in fnames]
        return fnames


    def load_single_day(path, header, ids=None):
        df = pd.read_csv(path, header=None)
        df = df.rename(columns = lambda i: header[i])
        df.drop(columns=[c for c in df.columns if 'Lane' in c] + ['District'], inplace=True)
        if ids:
            df = df[df['Station'].isin(ids)]
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Identify gaps in timelines
        num_gaps = 0
        all_timestamps = set(df['Timestamp'])
        interpolated = []
        groups = df.groupby('Station')
        for id, g in groups:
            if len(g) != len(g.dropna(subset=['Total Flow'])):
                _timestamps = set(g['Timestamp']).difference(set(g.dropna(subset=['Total Flow'])['Timestamp']))
                num_gaps += len(_timestamps)
                print(f'Found NaN in "Total Flow" at timestamps {_timestamps}')
                print('Interpolating...')

            diff = all_timestamps.difference(g['Timestamp'])
            if diff:
                num_gaps += len(diff)
                print(f'Missing observations ID {id} Timestamps: {diff}', file=sys.stderr)
                for elem in diff:
                    g = g.append({'Timestamp':elem}, ignore_index=True)

            g = g.sort_values('Timestamp')
            g = g.interpolate(method='ffill')
            g = g.fillna(method = 'pad')
            interpolated.append(g)

        df = pd.concat(interpolated)
        if num_gaps:
            print(f'Missing {num_gaps/len(df) * 100}% of the data')

        # Add derived time info
        #df['Year'] = df['Timestamp'].apply(lambda x: x.year)
        df['Day of week'] = df['Timestamp'].apply(lambda x: x.dayofweek)
        df['Month'] = df['Timestamp'].apply(lambda x: x.month)
        df['Day'] = df['Timestamp'].apply(lambda x: x.day)
        df['Hour'] = df['Timestamp'].apply(lambda x: x.hour)
        df['Minute'] = df['Timestamp'].apply(lambda x: x.minute)

        return df

    raw_paths = check_completeness(data_folder)
    for p in raw_paths:
       if p.endswith('.txt.gz'):
            unzip(p, p[:-3], data_folder)
    paths = [p[:-3] for p in raw_paths]

    # PEMS website doesn't provide headers in any of the files, so they have to be infered from the hints on site itself
    header = ['Timestamp', 'Station', 'District', 'Freeway #', 'Direction of Travel', 'Lane Type', 'Station Length',
              'Samples', '% Observed', 'Total Flow', 'Avg Occupancy', 'Avg Speed']
    header += [name.format(i) for i in range(8) for name in ['Lane {} Samples', 'Lane {} Flow', 'Lane {} Avg Occ', 'Lane {} Avg Speed', 'Lane {} Observed']]
    ids = [400001, 400017, 400030, 400040, 400045, 400052, 400057, 400059, 400065, 400069, 400073, 400084, 400085, 400088,
           400096, 400097, 400100, 400104, 400109, 400122, 400147, 400148, 400149, 400158, 400160, 400168, 400172, 400174,
           400178, 400185, 400201, 400206, 400209, 400213, 400221, 400222, 400227, 400236, 400238, 400240, 400246, 400253,
           400257, 400258, 400268, 400274, 400278, 400280, 400292, 400296, 400298, 400330, 400336, 400343, 400353, 400372,
           400394, 400400, 400414, 400418, 400429, 400435, 400436, 400440, 400449, 400457, 400461, 400464, 400479, 400485,
           400499, 400507, 400508, 400514, 400519, 400528, 400545, 400560, 400563, 400567, 400581, 400582, 400586, 400637,
           400643, 400648, 400649, 400654, 400664, 400665, 400668, 400673, 400677, 400687, 400688, 400690, 400700, 400709,
           400713, 400714, 400715, 400717, 400723, 400743, 400750, 400760, 400772, 400790, 400792, 400794, 400799, 400804,
           400822, 400823, 400828, 400832, 400837, 400842, 400863, 400869, 400873, 400895, 400904, 400907, 400911, 400916,
           400922, 400934, 400951, 400952, 400953, 400964, 400965, 400970, 400971, 400973, 400995, 400996, 401014, 401129,
           401154, 401163, 401167, 401210, 401224, 401327, 401351, 401388, 401391, 401400, 401403, 401440, 401457, 401464,
           401489, 401495, 401507, 401534, 401541, 401555, 401560, 401567, 401597, 401606, 401611, 401655, 401808, 401809,
           401810, 401811, 401816, 401817, 401845, 401846, 401890, 401891, 401906, 401908, 401926, 401936, 401937, 401942,
           401943, 401948, 401957, 401958, 401994, 401996, 401997, 401998, 402056, 402057, 402058, 402059, 402060, 402061,
           402067, 402117, 402118, 402119, 402120, 402121, 402281, 402282, 402283, 402284, 402285, 402286, 402287, 402288,
           402289, 402359, 402360, 402361, 402362, 402363, 402364, 402365, 402366, 402367, 402368, 402369, 402370, 402371,
           402372, 402373, 403225, 403265, 403329, 403401, 403402, 403404, 403406, 403409, 403412, 403414, 403419, 404370,
           404434, 404435, 404444, 404451, 404452, 404453, 404461, 404462, 404521, 404522, 404553, 404554, 404585, 404586,
           404640, 404753, 404759, 405613, 405619, 405701, 407150, 407151, 407152, 407153, 407155, 407157, 407161, 407165,
           407172, 407173, 407174, 407176, 407177, 407179, 407180, 407181, 407184, 407185, 407186, 407187, 407190, 407191,
           407194, 407200, 407202, 407204, 407206, 407207, 407321, 407323, 407325, 407328, 407331, 407332, 407335, 407336,
           407337, 407339, 407341, 407342, 407344, 407348, 407352, 407359, 407360, 407361, 407364, 407367, 407370, 407372,
           407373, 407374, 407710, 407711, 408907, 408911, 409524, 409525, 409526, 409528, 409529, 413026, 413845, 413877,
           413878, 414284, 414694]

    from tqdm import tqdm
    dfs = [load_single_day(p, header, ids) for p in tqdm(paths)]
    df = pd.concat(dfs)
    df['id'] = df['Station']
    df.reset_index(drop=True, inplace=True)
    df.to_csv(os.path.join(data_folder, 'pems_bay.csv'))
    print("Pems dataset created")
    # Construct graph
    print("Constructing graph")
    metafile= 'd04_text_meta_2017_01_04.txt'
    meta = pd.read_csv(os.path.join(data_folder, metafile), delimiter='\t', index_col='ID')
    meta = meta.loc[ids]
    nodes_loc = meta.loc[:,['Latitude', 'Longitude']].values
    graph = construct_graph(nodes_loc)
    normalized_loc = nodes_loc - nodes_loc.min(axis=0)
    normalized_loc /= normalized_loc.max(axis=0)
    graph.ndata['normalized_loc'] = torch.Tensor(normalized_loc) #Used for pretty printing
    pickle.dump(graph, open(os.path.join(data_folder, 'graph.bin'), 'wb'))

def main(args):
    """Runs main download routine.

    Args:
      expt_name: Name of experiment
      force_download: Whether to force data download from scratch
      output_folder: Folder path for storing data
    """

    print("#### Running download script ###")

    download_function = DOWNLOAD_FUNCTIONS[args.dataset]

    print("Getting {} data...".format(args.dataset))
    subdir = os.path.join(args.output_dir, args.dataset)
    print(subdir)
    if os.path.exists(subdir):
        print(f"Warning: Path {subdir} exists. Overwritting files!", file=sys.stderr)
    os.makedirs(subdir, exist_ok=True)
    download_function(subdir)

    print("Download completed.")

DOWNLOAD_FUNCTIONS = {
        "electricity": download_electricity,
        "traffic": download_traffic,
        "M5": download_m5,
        'pems_bay': download_PEMS_BAY,
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        type=str,
        choices=DOWNLOAD_FUNCTIONS.keys(),
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--output_dir",
        metavar="DIR",
        type=str,
        default=".",
        help="Path to folder for data download",
    )

    args = parser.parse_args()
    main(args)
