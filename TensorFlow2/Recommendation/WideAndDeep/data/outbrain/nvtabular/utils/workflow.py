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

import os
import shutil

import cudf
import cupy
import nvtabular as nvt
import rmm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from data.outbrain.nvtabular.utils.feature_description import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, \
    DISPLAY_ID_COLUMN, groupby_columns, ctr_columns
from nvtabular.io import Shuffle
from nvtabular.ops import Normalize, FillMedian, FillMissing, LogOp, LambdaOp, JoinGroupby, HashBucket
from nvtabular.ops.column_similarity import ColumnSimilarity
from nvtabular.utils import device_mem_size, get_rmm_size

TIMESTAMP_DELTA = 1465876799998


def get_devices():
    try:
        devices = [int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    except KeyError:
        from pynvml import nvmlInit, nvmlDeviceGetCount
        nvmlInit()
        devices = list(range(nvmlDeviceGetCount()))
    return devices


def _calculate_delta(col, gdf):
    col.loc[col == ''] = None
    col = col.astype('datetime64[ns]')
    timestamp = (gdf['timestamp'] + TIMESTAMP_DELTA).astype('datetime64[ms]')
    delta = (timestamp - col).dt.days
    delta = delta * (delta >= 0) * (delta <= 10 * 365)
    return delta


def _df_to_coo(df, row='document_id', col=None, data='confidence_level'):
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
            local_directory=local_directory
        )
        client = Client(cluster)
        setup_rmm_pool(client, device_pool_size)

    return client


def create_workflow(data_bucket_folder, output_bucket_folder, hash_spec, devices, local_directory):
    rmm.reinitialize(managed_memory=False)
    documents_categories_path = os.path.join(data_bucket_folder, 'documents_categories.csv')
    documents_topics_path = os.path.join(data_bucket_folder, 'documents_topics.csv')
    documents_entities_path = os.path.join(data_bucket_folder, 'documents_entities.csv')

    documents_categories_cudf = cudf.read_csv(documents_categories_path)
    documents_topics_cudf = cudf.read_csv(documents_topics_path)
    documents_entities_cudf = cudf.read_csv(documents_entities_path)
    documents_entities_cudf['entity_id'] = documents_entities_cudf['entity_id'].astype('category').cat.codes

    categories = _df_to_coo(documents_categories_cudf, col='category_id')
    topics = _df_to_coo(documents_topics_cudf, col='topic_id')
    entities = _df_to_coo(documents_entities_cudf, col='entity_id')

    del documents_categories_cudf, documents_topics_cudf, documents_entities_cudf
    ctr_thresh = {
        'ad_id': 5,
        'source_id_promo': 10,
        'publisher_id_promo': 10,
        'advertiser_id': 10,
        'campaign_id': 10,
        'document_id_promo': 5,

    }

    client = create_client(
        devices=devices,
        local_directory=local_directory
    )

    workflow = nvt.Workflow(
        cat_names=CATEGORICAL_COLUMNS,
        cont_names=CONTINUOUS_COLUMNS,
        label_name=['clicked'],
        client=client
    )

    workflow.add_feature([
        LambdaOp(
            op_name='country',
            f=lambda col, gdf: col.str.slice(0, 2),
            columns=['geo_location'], replace=False),
        LambdaOp(
            op_name='state',
            f=lambda col, gdf: col.str.slice(0, 5),
            columns=['geo_location'], replace=False),
        LambdaOp(
            op_name='days_since_published',
            f=_calculate_delta,
            columns=['publish_time', 'publish_time_promo'], replace=False),

        FillMedian(columns=['publish_time_days_since_published', 'publish_time_promo_days_since_published']),

        JoinGroupby(columns=['ad_id', 'source_id_promo', 'document_id_promo', 'publisher_id_promo', 'advertiser_id',
                             'campaign_id'],
                    cont_names=['clicked'], out_path=output_bucket_folder, stats=['sum', 'count']),
        LambdaOp(
            op_name='ctr',
            f=lambda col, gdf: ((col) / (gdf[col.name.replace('_clicked_sum', '_count')])).where(
                gdf[col.name.replace('_clicked_sum', '_count')] >= ctr_thresh[col.name.replace('_clicked_sum', '')], 0),
            columns=['ad_id_clicked_sum', 'source_id_promo_clicked_sum', 'document_id_promo_clicked_sum',
                     'publisher_id_promo_clicked_sum',
                     'advertiser_id_clicked_sum', 'campaign_id_clicked_sum'], replace=False),
        FillMissing(columns=groupby_columns + ctr_columns),
        LogOp(
            columns=groupby_columns + ['publish_time_days_since_published', 'publish_time_promo_days_since_published']),
        Normalize(columns=groupby_columns),
        ColumnSimilarity('doc_event_doc_ad_sim_categories', 'document_id', categories, 'document_id_promo',
                         metric='tfidf', on_device=False),
        ColumnSimilarity('doc_event_doc_ad_sim_topics', 'document_id', topics, 'document_id_promo', metric='tfidf',
                         on_device=False),
        ColumnSimilarity('doc_event_doc_ad_sim_entities', 'document_id', entities, 'document_id_promo', metric='tfidf',
                         on_device=False)
    ])

    workflow.add_cat_preprocess([
        HashBucket(hash_spec)
    ])
    workflow.finalize()

    return workflow


def create_parquets(data_bucket_folder, train_path, valid_path):
    cupy.random.seed(seed=0)
    rmm.reinitialize(managed_memory=True)
    documents_meta_path = os.path.join(data_bucket_folder, 'documents_meta.csv')
    clicks_train_path = os.path.join(data_bucket_folder, 'clicks_train.csv')
    events_path = os.path.join(data_bucket_folder, 'events.csv')
    promoted_content_path = os.path.join(data_bucket_folder, 'promoted_content.csv')

    documents_meta = cudf.read_csv(documents_meta_path, na_values=['\\N', ''])
    documents_meta = documents_meta.dropna(subset='source_id')
    documents_meta['publisher_id'].fillna(
        documents_meta['publisher_id'].isnull().cumsum() + documents_meta['publisher_id'].max() + 1, inplace=True)
    merged = (cudf.read_csv(clicks_train_path, na_values=['\\N', ''])
              .merge(cudf.read_csv(events_path, na_values=['\\N', '']), on=DISPLAY_ID_COLUMN, how='left',
                     suffixes=('', '_event'))
              .merge(cudf.read_csv(promoted_content_path, na_values=['\\N', '']), on='ad_id',
                     how='left',
                     suffixes=('', '_promo'))
              .merge(documents_meta, on='document_id', how='left')
              .merge(documents_meta, left_on='document_id_promo', right_on='document_id', how='left',
                     suffixes=('', '_promo')))
    merged['day_event'] = (merged['timestamp'] / 1000 / 60 / 60 / 24).astype(int)
    merged['platform'] = merged['platform'].fillna(1)
    merged['platform'] = merged['platform'] - 1
    display_event = merged[[DISPLAY_ID_COLUMN, 'day_event']].drop_duplicates().reset_index()
    random_state = cudf.Series(cupy.random.uniform(size=len(display_event)))
    valid_ids, train_ids = display_event.scatter_by_map(
        ((display_event.day_event <= 10) & (random_state > 0.2)).astype(int))
    valid_ids = valid_ids[DISPLAY_ID_COLUMN].drop_duplicates()
    train_ids = train_ids[DISPLAY_ID_COLUMN].drop_duplicates()
    valid_set = merged[merged[DISPLAY_ID_COLUMN].isin(valid_ids)]
    train_set = merged[merged[DISPLAY_ID_COLUMN].isin(train_ids)]
    valid_set = valid_set.sort_values(DISPLAY_ID_COLUMN)
    train_set.to_parquet(train_path, compression=None)
    valid_set.to_parquet(valid_path, compression=None)
    del merged, train_set, valid_set


def save_stats(data_bucket_folder, output_bucket_folder,
               output_train_folder, train_path, output_valid_folder,
               valid_path, stats_file, hash_spec, local_directory):
    devices = get_devices()
    shuffle = Shuffle.PER_PARTITION if len(devices) > 1 else True

    workflow = create_workflow(data_bucket_folder=data_bucket_folder,
                               output_bucket_folder=output_bucket_folder,
                               hash_spec=hash_spec,
                               devices=devices,
                               local_directory=local_directory)

    train_dataset = nvt.Dataset(train_path, part_mem_fraction=0.12)
    valid_dataset = nvt.Dataset(valid_path, part_mem_fraction=0.12)

    workflow.apply(train_dataset, record_stats=True, output_path=output_train_folder, shuffle=shuffle,
                   out_files_per_proc=5)
    workflow.apply(valid_dataset, record_stats=False, output_path=output_valid_folder, shuffle=None,
                   out_files_per_proc=None)

    workflow.save_stats(stats_file)

    return workflow


def clean(path):
    shutil.rmtree(path)


def execute_pipeline(config):
    required_folders = [config['temporary_folder'], config['output_train_folder'], config['output_valid_folder']]
    for folder in required_folders:
        os.makedirs(folder, exist_ok=True)

    create_parquets(
        data_bucket_folder=config['data_bucket_folder'],
        train_path=config['train_path'],
        valid_path=config['valid_path']
    )
    save_stats(
        data_bucket_folder=config['data_bucket_folder'],
        output_bucket_folder=config['output_bucket_folder'],
        output_train_folder=config['output_train_folder'],
        train_path=config['train_path'],
        output_valid_folder=config['output_valid_folder'],
        valid_path=config['valid_path'],
        stats_file=config['stats_file'],
        hash_spec=config['hash_spec'],
        local_directory=config['temporary_folder']
    )
    clean(config['temporary_folder'])
