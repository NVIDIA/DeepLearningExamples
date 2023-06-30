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
#
# author: Tomasz Grel (tgrel@nvidia.com)


import dataloading.feature_spec
import os
import numpy as np
import argparse

import dllogger

from dataloading.dataloader import create_input_pipelines
from nn.evaluator import Evaluator
from utils.logging import IterTimer, init_logging
import deployment.tf.triton_ensemble_wrapper
import deployment.hps.triton_ensemble_wrapper


def log_results(auc, test_loss, latencies, batch_size, compute_latencies=False, warmup_steps=10):
    # don't benchmark the first few warmup steps
    latencies = latencies[warmup_steps:]
    result_data = {
        'mean_inference_throughput': batch_size / np.mean(latencies),
        'mean_inference_latency': np.mean(latencies)
    }
    if compute_latencies:
        for percentile in [90, 95, 99]:
            result_data[f'p{percentile}_inference_latency'] = np.percentile(latencies, percentile)
    result_data['auc'] = auc
    result_data['test_loss'] = test_loss

    dllogger.log(data=result_data, step=tuple())


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path', type=str, required=True, help='')
    parser.add_argument('--dataset_type', default='tf_raw', type=str, help='')
    parser.add_argument('--feature_spec', default='feature_spec.yaml', type=str, help='')
    parser.add_argument('--batch_size', type=int, default=32768, help='Batch size')
    parser.add_argument('--auc_thresholds', type=int, default=8000, help='')

    parser.add_argument('--max_steps', type=int, default=None, help='')
    parser.add_argument('--print_freq', type=int, default=10, help='')

    parser.add_argument('--log_path', type=str, default='dlrm_tf_log.json', help='triton_inference_log.json')
    parser.add_argument('--verbose', action='store_true', default=False, help='')
    parser.add_argument('--test_on_train', action='store_true', default=False,
                        help='Run validation on the training set.')
    parser.add_argument('--fused_embedding', action='store_true', default=False,
                        help='Fuse the embedding table together for better GPU utilization.')
    parser.add_argument("--model_name", type=str, help="The name of the model used for inference.", required=True)

    parser.add_argument("--sparse_input_format", type=str, choices=["tf-savedmodel", "hps"],
                        required=True, default="tf-savedmodel")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_logging(log_path=args.log_path, params_dict=args.__dict__)
    fspec = dataloading.feature_spec.FeatureSpec.from_yaml(os.path.join(args.dataset_path, args.feature_spec))
    num_tables = len(fspec.get_categorical_sizes())
    table_ids = list(range(num_tables)) # possibly wrong ordering, to be tested

    train_pipeline, validation_pipeline = create_input_pipelines(dataset_type=args.dataset_type,
                                                    dataset_path=args.dataset_path,
                                                    train_batch_size=args.batch_size,
                                                    test_batch_size=args.batch_size,
                                                    table_ids=table_ids,
                                                    feature_spec=args.feature_spec,
                                                    rank=0, world_size=1)

    if args.test_on_train:
        validation_pipeline = train_pipeline

    if args.sparse_input_format == 'hps':
        wrapper_cls = deployment.hps.triton_ensemble_wrapper.RecsysTritonEnsemble
    else:
        wrapper_cls = deployment.tf.triton_ensemble_wrapper.RecsysTritonEnsemble

    model = wrapper_cls(model_name=args.model_name, num_tables=num_tables, verbose=args.verbose,
                        categorical_sizes=fspec.get_categorical_sizes(), fused_embedding=args.fused_embedding)

    timer = IterTimer(train_batch_size=args.batch_size, test_batch_size=args.batch_size,
                      optimizer=None, print_freq=args.print_freq, enabled=True)

    evaluator = Evaluator(model=model, timer=timer, auc_thresholds=args.auc_thresholds,
                          max_steps=args.max_steps, cast_dtype=None)

    auc, test_loss, latencies = evaluator(validation_pipeline=validation_pipeline)
    log_results(auc, test_loss, latencies, batch_size=args.batch_size)
    print('DONE')


if __name__ == '__main__':
    main()