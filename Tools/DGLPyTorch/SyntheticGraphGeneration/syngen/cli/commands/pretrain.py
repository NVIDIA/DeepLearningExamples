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

import argparse
import logging

from syngen.cli.commands.base_command import BaseCommand

from syngen.benchmark.tasks import train_ec

from syngen.configuration import SynGenDatasetFeatureSpec, SynGenConfiguration
from syngen.generator.tabular import tabular_generators_classes

from syngen.utils.types import MetaData
from syngen.benchmark.models import MODELS

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
log = logger


class PretrainCommand(BaseCommand):

    def init_parser(self, base_parser):
        pretrain_parser = base_parser.add_parser(
            "pretrain",
            help="Run Synthetic Graph Data Pre-training Tool",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        pretrain_parser.set_defaults(action=self.run)

        # global
        pretrain_parser.add_argument(
            "--task",
            type=str,
            default="ec",
            help=f"now the only available option is ec (edge-classification)",
        )
        pretrain_parser.add_argument(
            "--seed",
            type=int,
            default=777,
            help="Set a seed globally"
        )
        pretrain_parser.add_argument(
            "--timeit",
            action="store_true",
            help="Measures average training time",
        )
        pretrain_parser.add_argument(
            "--data-path",
            type=str,
            required=True,
            help="Path to dataset in SynGen format to train/finetune on",
        )
        pretrain_parser.add_argument(
            "--edge-name",
            type=str,
            required=True,
            help="Name of the edge to be used during train/finetune",
        )
        pretrain_parser.add_argument(
            "--pretraining-data-path",
            type=str,
            default=None,
            help="Path to dataset in SynGen format to pretrain on",
        )
        pretrain_parser.add_argument(
            "--pretraining-edge-name",
            type=str,
            default=None,
            help="Name of the edge to be used during pretraining",
        )

        # model
        pretrain_parser.add_argument(
            "--model",
            type=str,
            default="gat_ec",
            help=f"List of available models: {list(MODELS.keys())}",
        )
        pretrain_parser.add_argument(
            "--hidden-dim",
            type=int,
            default=128,
            help="Hidden feature dimension"
        )
        pretrain_parser.add_argument(
            "--out-dim",
            type=int,
            default=32,
            help="Output feature dimension",
        )
        pretrain_parser.add_argument(
            "--num-classes",
            type=int,
            required=True,
            help="Number of classes in the target column",
        )
        pretrain_parser.add_argument(
            "--n-layers",
            type=int,
            default=1,
            help="Multi-layer full neighborhood sampler layers",
        )
        for key in MODELS.keys():
            MODELS[key].add_args(pretrain_parser)

        # dataset
        pretrain_parser.add_argument(
            "--target-col",
            type=str,
            required=True,
            help="Target column for downstream prediction",
        )
        pretrain_parser.add_argument(
            "--train-ratio",
            type=float,
            default=0.8,
            help="Ratio of data to use as train",
        )
        pretrain_parser.add_argument(
            "--val-ratio",
            type=float,
            default=0.1,
            help="Ratio of data to use as val",
        )
        pretrain_parser.add_argument(
            "--test-ratio",
            type=float,
            default=0.1,
            help="Ratio of data to use as test",
        )

        # training
        pretrain_parser.add_argument(
            "--learning-rate",
            "--lr",
            dest="learning_rate",
            type=float,
            default=1e-3,
            help=f"Initial learning rate for optimizer",
        )
        pretrain_parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.1,
            help=f"Weight decay for optimizer",
        )
        pretrain_parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="Pre-training and Fine-tuning dataloader batch size",
        )
        pretrain_parser.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help="Number of dataloading workers",
        )
        pretrain_parser.add_argument(
            "--shuffle",
            action="store_true",
            default=False,
            help="Shuffles data each epoch"
        )
        pretrain_parser.add_argument(
            "--pretrain-epochs",
            type=int,
            default=0,
            help="Number of pre-training epochs",
        )
        pretrain_parser.add_argument(
            "--finetune-epochs",
            type=int,
            default=1,
            help="Number of finetuning epochs",
        )
        pretrain_parser.add_argument(
            "--log-interval",
            type=int,
            default=1,
            help="logging interval"
        )

    def run(self, args):
        dict_args = vars(args)

        finetune_feature_spec = SynGenDatasetFeatureSpec.instantiate_from_preprocessed(
            dict_args['data_path']
        )
        pretrain_feature_spec = None

        if dict_args['pretraining_data_path']:
            pretrain_feature_spec = SynGenDatasetFeatureSpec.instantiate_from_preprocessed(
                dict_args['pretraining_data_path']
            )

        if args.task == "ec":
            out = train_ec(
                args,
                finetune_feature_spec=finetune_feature_spec,
                pretrain_feature_spec=pretrain_feature_spec,
            )
        else:
            raise ValueError("benchmark not supported")
        log.info(out)
        return out
