# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing script for SIM models."""
import logging
import multiprocessing
import os

import click
import cudf
import cupy
import dask.dataframe
import dask_cudf
import rmm

from preprocessing.io import load_metadata, load_review_data, save_metadata
from preprocessing.ops import ExplodeSequence, add_negative_sequence, list_slice, slice_and_pad_left

DASK_TRAIN_DATASET_CHUNKSIZE = 15_000
TRAIN_DATA_DIR = "train"
TEST_DATA_DIR = "test"
TEST_DATA_FILE = "part.0.parquet"
CATEGORIZED_METADATA_FILE = "metadata.json"
OUTPUT_META = {
    "label": "int8",
    "uid": "int64",
    "item": "int32",
    "cat": "int32",
    "item_sequence": "list",
    "cat_sequence": "list",
    "neg_item_sequence": "list",
    "neg_cat_sequence": "list",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def add_categorified_column(df, col_name, id_col_name):
    unique_values = df[col_name].unique().to_frame()
    unique_values[id_col_name] = cupy.arange(len(unique_values), dtype="int32") + 1
    df = df.merge(unique_values, how="left", on=col_name)
    return df


def categorify_items(all_items_unique: cudf.DataFrame, metadata: cudf.DataFrame) -> cudf.DataFrame:
    unique_item_with_category = all_items_unique.merge(metadata, how="left", on="item")
    unique_item_with_category = unique_item_with_category.fillna("no_category")

    df = add_categorified_column(unique_item_with_category, "item", "item_id")
    df = add_categorified_column(df, "cat", "cat_id")

    return df


def filter_too_short_sequences(reviews: cudf.DataFrame, min_seq_length: int) -> cudf.DataFrame:
    user_counts = reviews["user"].value_counts()
    user_counts_filtered = user_counts[user_counts >= min_seq_length]
    valid_users = user_counts_filtered.index

    reviews = reviews[reviews["user"].isin(valid_users)]
    reviews.reset_index(drop=True, inplace=True)

    return reviews


def add_items_and_categories_indices(
    reviews: cudf.DataFrame,
    item_and_cat_with_ids: cudf.DataFrame,
) -> cudf.DataFrame:
    return reviews.merge(item_and_cat_with_ids, how="left", on="item")


def categorify_users(reviews: cudf.DataFrame) -> cudf.DataFrame:
    return add_categorified_column(reviews, "user", "uid")


def create_sampling_df(
    all_items: cudf.DataFrame,
    item_and_cat_with_ids: cudf.DataFrame
) -> cudf.DataFrame:
    sampling_df = all_items.merge(item_and_cat_with_ids, how="left", on="item")
    sampling_df = sampling_df[["item_id", "cat_id"]]
    sampling_df = sampling_df.sort_values(by="item_id")
    sampling_df.reset_index(drop=True, inplace=True)
    return sampling_df


def aggregate_per_user(df):
    df = df.sort_values(by=["unixReviewTime", "item"])
    df = df.groupby("uid").agg({
        "item_id": list,
        "cat_id": list,
    })
    df.reset_index(inplace=True)
    df = df.rename(columns={
        "item_id": "item_sequence",
        "cat_id": "cat_sequence",
    })

    df["item"] = df["item_sequence"].list.get(-1)
    df["cat"] = df["cat_sequence"].list.get(-1)

    df["item_sequence"] = list_slice(df["item_sequence"], 0, -1)
    df["cat_sequence"] = list_slice(df["cat_sequence"], 0, -1)

    return df


def explode_sequence(df: cudf.DataFrame, min_elements: int, max_elements: int) -> cudf.DataFrame:
    df = ExplodeSequence(
        col_names=["item_sequence", "cat_sequence"],
        keep_cols=["uid"],
        max_elements=max_elements + 1,
    ).transform(df)

    df["item"] = df["item_sequence"].list.get(-1)
    df["cat"] = df["cat_sequence"].list.get(-1)

    df["item_sequence"] = list_slice(df["item_sequence"], 0, -1)
    df["cat_sequence"] = list_slice(df["cat_sequence"], 0, -1)

    df = df[df.item_sequence.list.len() >= min_elements]

    return df


def add_negative_label(pos_df: cudf.DataFrame, sampling_df: cudf.DataFrame) -> cudf.DataFrame:
    neg_df = pos_df.copy()
    pos_df["label"] = cupy.int8(1)
    neg_df["label"] = cupy.int8(0)

    neg = cupy.random.randint(
        low=0,
        high=len(sampling_df),
        size=len(neg_df),
        dtype=int,
    )

    neg_item_ids = sampling_df["item_id"].iloc[neg].values
    neg_df["item"] = neg_item_ids

    neg_cat_ids = sampling_df["cat_id"].iloc[neg].values
    neg_df["cat"] = neg_cat_ids

    df = cudf.concat([pos_df, neg_df])

    return df


def add_negative_sampling(df: cudf.DataFrame, sampling_df: cudf.DataFrame) -> cudf.DataFrame:
    df = add_negative_label(df, sampling_df)

    neg = cupy.random.randint(
        low=0,
        high=len(sampling_df),
        size=int(df.item_sequence.list.len().sum()),
        dtype=int,
    )
    item_samples = sampling_df["item_id"].iloc[neg]
    cat_samples = sampling_df["cat_id"].iloc[neg]

    df["neg_item_sequence"] = add_negative_sequence(df["item_sequence"], item_samples)
    df["neg_cat_sequence"] = add_negative_sequence(df["cat_sequence"], cat_samples)

    return df


def pad_with_zeros(df: cudf.DataFrame, max_elements: int) -> cudf.DataFrame:
    df["item_sequence"] = slice_and_pad_left(df["item_sequence"], max_elements)
    df["cat_sequence"] = slice_and_pad_left(df["cat_sequence"], max_elements)
    df["neg_item_sequence"] = slice_and_pad_left(df["neg_item_sequence"], max_elements)
    df["neg_cat_sequence"] = slice_and_pad_left(df["neg_cat_sequence"], max_elements)

    return df


def create_train_dataset(
    df: cudf.DataFrame,
    sampling_df: cudf.DataFrame,
    min_elements: int,
    max_elements: int,
    output_path: str,
    seed: int,
    dask_scheduler: str = "processes",
) -> None:
    def transform(df, sampling_df, partition_info):
        part_seed = seed + partition_info["number"] + 1
        cupy.random.seed(part_seed)

        df = explode_sequence(df, min_elements, max_elements)
        df = add_negative_sampling(df, sampling_df)
        df = pad_with_zeros(df, max_elements)
        df = df.sort_values(by=["uid"])
        df.reset_index(drop=True, inplace=True)
        df = df[list(OUTPUT_META)]
        return df

    ddf = dask_cudf.from_cudf(df, chunksize=DASK_TRAIN_DATASET_CHUNKSIZE)
    ddf = ddf.map_partitions(transform, meta=OUTPUT_META, sampling_df=sampling_df)
    ddf = ddf.clear_divisions()

    with dask.config.set(scheduler=dask_scheduler):
        ddf.to_parquet(output_path, write_index=False, overwrite=True)


def create_test_dataset(
    df: cudf.DataFrame,
    sampling_df: cudf.DataFrame,
    max_elements: int,
    output_path: str,
) -> None:
    df = add_negative_sampling(df, sampling_df)
    df = pad_with_zeros(df, max_elements)
    df = df.sort_values(by=["uid"])
    df.reset_index(drop=True, inplace=True)
    df = df[list(OUTPUT_META)]

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, TEST_DATA_FILE)
    df.to_parquet(output_file, index=False)


@click.command()
@click.option(
    "--amazon_dataset_path",
    required=True,
    help="Path to the dataset. Must contain both reviews and metadata json files.",
    type=str,
)
@click.option(
    "--output_path",
    required=True,
    help="Path where preprocessed dataset is saved.",
    type=str,
)
@click.option(
    "--metadata_file_name",
    default="meta_Books.json",
    help="Path to the dataset. Must contain both reviews and metadata json files.",
    type=str,
)
@click.option(
    "--reviews_file_name",
    default="reviews_Books.json",
    help="Path where preprocessed dataset is saved.",
    type=str,
)
@click.option(
    "--max_sequence_length",
    default=100,
    help="Take only `max_sequence_length` last elements of a sequence.",
)
@click.option(
    "--shortest_sequence_for_user",
    default=20,
    help="Specifies what is a minimal length of a sequence. "
    "Every user with a sequence shorter than this value will be discarded."
)
@click.option(
    "--shortest_sequence_for_training",
    default=1,
    help="Specifies what is a minimal length of a sequence in a training set.",
)
@click.option(
    "--metadata_loader_n_proc",
    default=multiprocessing.cpu_count(),
    help="Specifies the number of processes used to parse metadata.",
)
@click.option(
    "--review_loader_num_workers",
    default=20,
    help="Specifies the number of dask workers used to read reviews data. "
    "Note that, as each worker is a new process, too high value might cause GPU OOM errors."
)
@click.option(
    "--seed",
    default=12345,
    help="Seed for reproducibility."
    "Note that the results can still differ between machines because of dask/cudf non-determinism.",
    type=int,
)
def main(
    amazon_dataset_path: str,
    output_path: str,
    metadata_file_name: str,
    reviews_file_name: str,
    max_sequence_length: int,
    shortest_sequence_for_user: int,
    shortest_sequence_for_training: int,
    metadata_loader_n_proc: int,
    review_loader_num_workers: int,
    seed: int,
):
    cupy.random.seed(seed)
    rmm.reinitialize(managed_memory=True)

    metadata_path = os.path.join(amazon_dataset_path, metadata_file_name)
    reviews_path = os.path.join(amazon_dataset_path, reviews_file_name)

    logging.info("Loading metadata")
    metadata = load_metadata(metadata_path, metadata_loader_n_proc)
    assert len(metadata) == metadata["item"].nunique(), "metadata should contain unique items"

    logging.info("Loading review data")
    reviews = load_review_data(reviews_path, review_loader_num_workers)

    logging.info("Removing short user sequences")
    reviews = filter_too_short_sequences(reviews, shortest_sequence_for_user)

    logging.info("Categorifying users, items, categories")
    all_items_unique = reviews["item"].unique().to_frame()
    item_and_cat_with_ids = categorify_items(all_items_unique, metadata)
    reviews = add_items_and_categories_indices(reviews, item_and_cat_with_ids)
    reviews = categorify_users(reviews)

    logging.info("Aggregating data per user")
    df = aggregate_per_user(reviews)

    logging.info("Preparing dataframe for negative sampling")
    all_items = reviews["item"].to_frame()
    sampling_df = create_sampling_df(all_items, item_and_cat_with_ids)

    os.makedirs(output_path, exist_ok=True)

    logging.info("Creating train dataset")
    create_train_dataset(
        df,
        sampling_df,
        min_elements=shortest_sequence_for_training,
        max_elements=max_sequence_length,
        output_path=os.path.join(output_path, TRAIN_DATA_DIR),
        seed=seed,
    )

    logging.info("Creating test dataset")
    create_test_dataset(
        df,
        sampling_df,
        max_elements=max_sequence_length,
        output_path=os.path.join(output_path, TEST_DATA_DIR),
    )

    logging.info("Saving metadata")
    save_metadata(
        number_of_items=len(item_and_cat_with_ids),
        number_of_categories=item_and_cat_with_ids["cat_id"].nunique(),
        number_of_users=len(df),
        output_path=os.path.join(output_path, CATEGORIZED_METADATA_FILE),
    )


if __name__ == "__main__":
    main()
