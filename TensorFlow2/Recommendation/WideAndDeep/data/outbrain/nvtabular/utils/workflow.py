# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import shutil

import cudf
import cupy
import numpy as np
import nvtabular as nvt
import rmm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from data.outbrain.features import get_features_keys
from data.outbrain.nvtabular.utils.feature_description import (
    CATEGORICAL_COLUMNS, CTR_INPUTS, DISPLAY_ID_COLUMN)
from nvtabular import ColumnGroup
from nvtabular.io import Shuffle
from nvtabular.ops import (Categorify, ColumnSelector, FillMedian, FillMissing,
                           HashBucket, JoinExternal, JoinGroupby, LambdaOp,
                           ListSlice, LogOp, Normalize, Operator, Rename)
from nvtabular.ops.column_similarity import ColumnSimilarity
from nvtabular.utils import device_mem_size, get_rmm_size

TIMESTAMP_DELTA = 1465876799998


def get_devices():
    try:
        devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        ]
    except KeyError:
        from pynvml import nvmlDeviceGetCount, nvmlInit

        nvmlInit()
        devices = list(range(nvmlDeviceGetCount()))
    return devices


class DaysSincePublished(Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            col.loc[col == ""] = None
            col = col.astype("datetime64[ns]")
            timestamp = (gdf["timestamp"] + TIMESTAMP_DELTA).astype("datetime64[ms]")
            delta = (timestamp - col).dt.days
            gdf[column + "_days_since_published"] = (
                    delta * (delta >= 0) * (delta <= 10 * 365)
            )
        return gdf

    def output_column_names(self, columns):
        return ColumnSelector(
            [column + "_days_since_published" for column in columns.names]
        )

    def dependencies(self):
        return ["timestamp"]


def _df_to_coo(df, row="document_id", col=None, data="confidence_level"):
    return cupy.sparse.coo_matrix((df[data].values, (df[row].values, df[col].values)))


def setup_rmm_pool(client, pool_size):
    pool_size = get_rmm_size(pool_size)
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=pool_size)
    return None


def create_client(devices, local_directory):
    client = None

    if len(devices) > 1:
        device_size = device_mem_size(kind="total")
        device_limit = int(0.8 * device_size)
        device_pool_size = int(0.8 * device_size)
        cluster = LocalCUDACluster(
            n_workers=len(devices),
            CUDA_VISIBLE_DEVICES=",".join(str(x) for x in devices),
            device_memory_limit=device_limit,
            local_directory=local_directory,
        )
        client = Client(cluster)
        setup_rmm_pool(client, device_pool_size)

    return client


def create_workflow(data_bucket_folder, hash_spec, devices, local_directory, dask):
    rmm.reinitialize(managed_memory=False)

    documents_categories_path = os.path.join(
        data_bucket_folder, "documents_categories.csv"
    )
    documents_topics_path = os.path.join(data_bucket_folder, "documents_topics.csv")
    documents_entities_path = os.path.join(data_bucket_folder, "documents_entities.csv")

    documents_categories_cudf = cudf.read_csv(documents_categories_path)
    documents_topics_cudf = cudf.read_csv(documents_topics_path)
    documents_entities_cudf = cudf.read_csv(documents_entities_path)
    documents_entities_cudf["entity_id"] = (
        documents_entities_cudf["entity_id"].astype("category").cat.codes
    )
    documents_categories_grouped = (
        documents_categories_cudf.groupby("document_id")
            .agg({"category_id": "collect", "confidence_level": "collect"})
            .reset_index()
    )
    documents_categories_grouped = documents_categories_grouped.rename(
        columns={
            "category_id": "category_id_list",
            "confidence_level": "confidence_level_cat_list",
        }
    )
    documents_entities_grouped = (
        documents_entities_cudf.groupby("document_id")
            .agg({"entity_id": "collect", "confidence_level": "collect"})
            .reset_index()
    )
    documents_entities_grouped = documents_entities_grouped.rename(
        columns={
            "entity_id": "entity_id_list",
            "confidence_level": "confidence_level_ent_list",
        }
    )
    documents_topics_grouped = (
        documents_topics_cudf.groupby("document_id")
            .agg({"topic_id": "collect", "confidence_level": "collect"})
            .reset_index()
    )
    documents_topics_grouped = documents_topics_grouped.rename(
        columns={
            "topic_id": "topic_id_list",
            "confidence_level": "confidence_level_top_list",
        }
    )

    categories = _df_to_coo(documents_categories_cudf, col="category_id")
    topics = _df_to_coo(documents_topics_cudf, col="topic_id")
    entities = _df_to_coo(documents_entities_cudf, col="entity_id")

    del documents_categories_cudf, documents_topics_cudf, documents_entities_cudf
    ctr_thresh = {
        "ad_id": 5,
        "source_id_promo": 10,
        "publisher_id_promo": 10,
        "advertiser_id": 10,
        "campaign_id": 10,
        "document_id_promo": 5,
    }

    cat_cols = ColumnGroup(CATEGORICAL_COLUMNS)

    def get_slice(num_char):
        def lambda_slice(col, gdf):
            return col.str.slice(0, num_char)

        return lambda_slice

    geo_location = ColumnGroup(["geo_location"])
    country = geo_location >> LambdaOp(get_slice(2)) >> Rename(postfix="_country")
    state = geo_location >> LambdaOp(get_slice(5)) >> Rename(postfix="_state")
    geo_features = geo_location + country + state

    dates = ["publish_time", "publish_time_promo"]
    date_features = dates >> DaysSincePublished() >> FillMedian() >> LogOp

    ctr_inputs = ColumnGroup(CTR_INPUTS)

    stat_cols = ctr_inputs >> JoinGroupby(cont_cols=["clicked"], stats=["sum", "count"])

    def calculate_ctr_with_filter(col, gdf):
        col = col.astype(np.float32)
        ctr_col_name = col.name.replace("_clicked_sum", "")
        ctr_count_name = col.name.replace("_clicked_sum", "_count")

        col = col / gdf[ctr_count_name]  # CTR
        col = col.where(gdf[ctr_count_name] >= ctr_thresh[ctr_col_name], 0)  # Filter

        return col

    ctr_selected_features = [column + "_clicked_sum" for column in ctr_inputs.names]
    dependency_features = [column + "_count" for column in ctr_inputs.names]

    ctr_cols = (
            stat_cols[ctr_selected_features]
            >> LambdaOp(
        calculate_ctr_with_filter, dependency=stat_cols[dependency_features]
    )
            >> Rename(f=lambda x: x.replace("_clicked_sum", "_ctr"))
    )

    stat_cols = stat_cols >> FillMissing() >> LogOp() >> Normalize()

    ctr_cols = ctr_cols >> FillMissing()

    cat_cols = cat_cols + geo_features >> HashBucket(dict(list(hash_spec.items())[:-3]))

    sim_features_categories = (
            [["document_id", "document_id_promo"]]
            >> ColumnSimilarity(categories, metric="tfidf", on_device=False)
            >> Rename(postfix="_categories")
    )
    sim_features_topics = (
            [["document_id", "document_id_promo"]]
            >> ColumnSimilarity(topics, metric="tfidf", on_device=False)
            >> Rename(postfix="_topics")
    )
    sim_features_entities = (
            [["document_id", "document_id_promo"]]
            >> ColumnSimilarity(entities, metric="tfidf", on_device=False)
            >> Rename(postfix="_entities")
    )

    sim_features = sim_features_categories + sim_features_topics + sim_features_entities

    joined = ["document_id"] >> JoinExternal(
        documents_categories_grouped,
        on=["document_id"],
        on_ext=["document_id"],
        how="left",
        columns_ext=["category_id_list", "confidence_level_cat_list", "document_id"],
        cache="device",
    )

    joined = joined >> JoinExternal(
        documents_entities_grouped,
        on=["document_id"],
        on_ext=["document_id"],
        how="left",
        columns_ext=["entity_id_list", "confidence_level_ent_list", "document_id"],
        cache="device",
    )
    joined = joined >> JoinExternal(
        documents_topics_grouped,
        on=["document_id"],
        on_ext=["document_id"],
        how="left",
        columns_ext=["topic_id_list", "confidence_level_top_list", "document_id"],
        cache="device",
    )

    categorified_multihots = (
            joined[["topic_id_list", "entity_id_list", "category_id_list"]]
            >> Categorify()
            >> FillMissing()
            >> ListSlice(3)
            >> HashBucket(dict(list(hash_spec.items())[-3:]))
    )

    features = (
            date_features
            + ctr_cols
            + stat_cols
            + cat_cols
            + sim_features
            + categorified_multihots
            + ["clicked", "display_id"]
    )

    client = (
        create_client(devices=devices, local_directory=local_directory)
        if dask
        else None
    )
    required_features = get_features_keys() + ["clicked"]

    workflow = nvt.Workflow(features[required_features], client=client)

    return workflow


def create_parquets(data_bucket_folder, train_path, valid_path):
    cupy.random.seed(seed=0)
    rmm.reinitialize(managed_memory=True)
    documents_meta_path = os.path.join(data_bucket_folder, "documents_meta.csv")
    clicks_train_path = os.path.join(data_bucket_folder, "clicks_train.csv")
    events_path = os.path.join(data_bucket_folder, "events.csv")
    promoted_content_path = os.path.join(data_bucket_folder, "promoted_content.csv")
    documents_meta = cudf.read_csv(documents_meta_path, na_values=["\\N", ""])
    documents_meta["publisher_id"].fillna(
        documents_meta["publisher_id"].isnull().cumsum()
        + documents_meta["publisher_id"].max()
        + 1,
        inplace=True,
    )
    merged = (
        cudf.read_csv(clicks_train_path, na_values=["\\N", ""])
            .merge(
            cudf.read_csv(events_path, na_values=["\\N", ""]),
            on=DISPLAY_ID_COLUMN,
            how="left",
            suffixes=("", "_event"),
        )
            .merge(
            cudf.read_csv(promoted_content_path, na_values=["\\N", ""]),
            on="ad_id",
            how="left",
            suffixes=("", "_promo"),
        )
            .merge(documents_meta, on="document_id", how="left")
            .merge(
            documents_meta,
            left_on="document_id_promo",
            right_on="document_id",
            how="left",
            suffixes=("", "_promo"),
        )
    )
    merged["day_event"] = (merged["timestamp"] / 1000 / 60 / 60 / 24).astype(int)
    merged["platform"] = merged["platform"].fillna(1)
    merged["platform"] = merged["platform"] - 1
    display_event = (
        merged[[DISPLAY_ID_COLUMN, "day_event"]].drop_duplicates().reset_index()
    )
    random_state = cudf.Series(cupy.random.uniform(size=len(display_event)))
    valid_ids, train_ids = display_event.scatter_by_map(
        ((display_event.day_event <= 10) & (random_state > 0.2)).astype(int)
    )
    valid_ids = valid_ids[DISPLAY_ID_COLUMN].drop_duplicates()
    train_ids = train_ids[DISPLAY_ID_COLUMN].drop_duplicates()
    valid_set = merged[merged[DISPLAY_ID_COLUMN].isin(valid_ids)]
    train_set = merged[merged[DISPLAY_ID_COLUMN].isin(train_ids)]
    valid_set = valid_set.sort_values(DISPLAY_ID_COLUMN)
    train_set.to_parquet(train_path, compression=None)
    valid_set.to_parquet(valid_path, compression=None)
    del merged, train_set, valid_set


def save_stats(
        data_bucket_folder,
        output_train_folder,
        train_path,
        output_valid_folder,
        valid_path,
        stats_file,
        hash_spec,
        local_directory,
        dask,
):
    devices = get_devices()
    shuffle = Shuffle.PER_PARTITION if len(devices) > 1 else True

    workflow = create_workflow(
        data_bucket_folder=data_bucket_folder,
        hash_spec=hash_spec,
        devices=devices,
        local_directory=local_directory,
        dask=dask,
    )

    train_dataset = nvt.Dataset(train_path, part_size="150MB")
    valid_dataset = nvt.Dataset(valid_path, part_size="150MB")
    workflow.fit(train_dataset)
    workflow.transform(train_dataset).to_parquet(
        output_path=output_train_folder, shuffle=shuffle, out_files_per_proc=8
    )
    workflow.transform(valid_dataset).to_parquet(
        output_path=output_valid_folder, shuffle=None, output_files=8
    )

    workflow.save(stats_file)

    return workflow


def clean(path):
    shutil.rmtree(path)


def execute_pipeline(config):
    required_folders = [
        config["temporary_folder"],
        config["output_train_folder"],
        config["output_valid_folder"],
    ]
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)

    create_parquets(
        data_bucket_folder=config["data_bucket_folder"],
        train_path=config["train_path"],
        valid_path=config["valid_path"],
    )
    save_stats(
        data_bucket_folder=config["data_bucket_folder"],
        output_train_folder=config["output_train_folder"],
        train_path=config["train_path"],
        output_valid_folder=config["output_valid_folder"],
        valid_path=config["valid_path"],
        stats_file=config["stats_file"],
        hash_spec=config["hash_spec"],
        local_directory=config["temporary_folder"],
        dask=config["dask"],
    )

    clean(config["temporary_folder"])
    clean("./categories")
