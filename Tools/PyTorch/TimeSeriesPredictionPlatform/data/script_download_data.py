# Copyright 2021-2022 NVIDIA Corporation

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


def unzip(zip_path, output_file, data_folder):
    """Unzips files and checks successful completion."""

    print("Unzipping file: {}".format(zip_path))
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

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"

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
