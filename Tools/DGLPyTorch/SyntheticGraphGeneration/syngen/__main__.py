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
import os
import pathlib
import re
import sys

import pandas as pd

from typing import Union
from pathlib import Path

from syngen.benchmark.models import MODELS
from syngen.benchmark.tasks import train_ec
from syngen.generator.graph import BaseGenerator as BaseGraphGenerator
from syngen.generator.tabular import BaseTabularGenerator
from syngen.graph_aligner import BaseGraphAligner
from syngen.preprocessing.datasets import DATASETS
from syngen.synthesizer import BaseSynthesizer
from syngen.utils import dynamic_import
from syngen.utils.types import MetaData

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
log = logger


def _run_synthesizer(args):
    data = {}
    dict_args = vars(args)
    data_path = args.data_path

    synthesizers = _get_synthesizers()
    synthesizer = synthesizers[args.synthesizer]
    preprocessing = getattr(args, "preprocessing", None)

    if preprocessing is not None:
        preprocessing = DATASETS[preprocessing](**dict_args)
        data = preprocessing.transform(data_path)
    graph_info = getattr(preprocessing, "graph_info", None)
    edge_data = data.get(MetaData.EDGE_DATA, None)
    node_data = data.get(MetaData.NODE_DATA, None)

    if edge_data is not None:  # - data ingestible synthesizers
        aligners = _get_aligners()
        aligner = aligners.get(args.aligner, None)
        aligner = aligner(**dict_args)
        tabular_generators = _get_tabular_generators()
        edge_generator = tabular_generators.get(args.edge_generator, None)
        dict_args.pop("edge_generator")
        node_generator = tabular_generators.get(args.node_generator, None)
        dict_args.pop("node_generator")
        if edge_generator is not None:
            edge_generator = edge_generator(
                batch_size=args.eg_batch_size,
                epochs=args.eg_epochs,
                **dict_args,
            )
        if node_generator is not None:
            node_generator = node_generator(
                batch_size=args.ng_batch_size,
                epochs=args.ng_epochs,
                **dict_args,
            )

        graph_generators = _get_graph_generators()
        graph_generator = graph_generators.get(args.graph_generator)(
            seed=args.gg_seed, **dict_args
        )
        dict_args.pop("graph_generator")

        synthesizer = synthesizer(
            graph_generator=graph_generator,
            graph_info=graph_info,
            edge_feature_generator=edge_generator,
            node_feature_generator=node_generator,
            graph_aligner=aligner,
            **dict_args,
        )
        synthesizer.fit(edge_data=edge_data, node_data=node_data, **dict_args)
    else:  # - dataless synthesizers
        edge_dim = getattr(args, "edge_dim", None)
        node_dim = getattr(args, "node_dim", None)
        is_directed = getattr(args, "g_directed", None)
        bipartite = getattr(args, "g_bipartite", None)
        synthesizer = synthesizer(is_directed=is_directed, bipartite=bipartite, **dict_args)
        synthesizer.fit(edge_dim=edge_dim, node_dim=node_dim)

    data = synthesizer.generate(
        num_nodes=getattr(args, "num_nodes", None),
        num_edges=getattr(args, "num_edges", None),
        num_nodes_src_set=getattr(args, "num_nodes_src_set", None),
        num_nodes_dst_set=getattr(args, "num_nodes_dst_set", None),
        num_edges_src_dst=getattr(args, "num_edges_src_dst", None),
        num_edges_dst_src=getattr(args, "num_edges_dst_src", None),
    )
    synthesizer.cleanup_session()
    log.info("Done synthesizing dataset...")
    return data


def _run_pretraining(args):
    data_pretrain = None
    data_finetune = None
    dict_args = vars(args)
    data_path = args.data_path
    if args.pretraining_path is not None:
        data_pretrain = {}
        generated_data_path = Path(args.pretraining_path)

        files = os.listdir(generated_data_path)
        for fname in files:
            if MetaData.EDGE_DATA.value in fname:
                df = pd.read_csv(generated_data_path / fname)
                data_pretrain[MetaData.EDGE_DATA] = df
            elif MetaData.NODE_DATA.value in fname:
                df = pd.read_csv(generated_data_path / fname)
                data_pretrain[MetaData.NODE_DATA] = df

    preprocessing = getattr(args, "preprocessing")
    preprocessing = DATASETS[preprocessing](**dict_args)
    data_finetune = preprocessing.transform(data_path)

    if args.task == "ec":
        out = train_ec(
            args,
            pretrain_source=data_pretrain,
            graph_info=preprocessing.graph_info,
            finetune_source=data_finetune,
        )
    else:
        raise ValueError("benchmark not supproted")
    log.info(out)
    return out


def _cap_str_cammel_case_str(s):
    i = 0
    l = len(s)
    p = []
    for j in range(1, l):
        if s[j].isupper() and j + 1 < l and s[j + 1].islower():
            p.append(s[i:j].lower())
            i = j
    p.append(s[i:].lower())
    return "_".join(p)


def _get_objects(obj_path_dict):
    obj_dict = {}
    for obj_name, path in obj_path_dict.items():
        obj_name = _cap_str_cammel_case_str(obj_name)
        obj_dict[obj_name] = dynamic_import(path)
    return obj_dict


def _get_synthesizers():
    synthesizer_re = re.compile("(\s*)synthesizer(\s*)")
    objs = _get_objects(BaseSynthesizer.get_synthesizers())
    keys = list(objs.keys())
    for k in keys:
        key = synthesizer_re.sub("", k)
        objs[key.strip("_")] = objs.pop(k)
    return objs


def _get_aligners():
    align_re = re.compile("(\s*)aligner(\s*)")
    objs = _get_objects(BaseGraphAligner.get_aligners())
    keys = list(objs.keys())
    for k in keys:
        key = align_re.sub("", k)
        objs[key.strip("_")] = objs.pop(k)
    return objs


def _get_graph_generators():
    objs = _get_objects(BaseGraphGenerator.get_generators())
    keys = list(objs.keys())
    for k in keys:
        objs[k.split("generator")[0].strip("_")] = objs.pop(k)
    return objs


def _get_tabular_generators():
    objs = _get_objects(BaseTabularGenerator.get_generators())
    gen_re = re.compile("(\s*)generator(\s*)")
    keys = list(objs.keys())
    for k in keys:
        key = gen_re.sub("", k)
        objs[key.strip("_")] = objs.pop(k)
    return objs


def list_available_tabular_generators():
    return [k for k in _get_tabular_generators()]


def list_available_graph_generators():
    return [k for k in _get_graph_generators()]


def list_available_graph_aligner():
    return [k for k in _get_aligners()]


def list_available_synthesizer():
    return [k for k in _get_synthesizers()]


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Synthetic Graph Generation Tool"
    )
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title="action")
    action.required = True

    # - synthesizer
    synthesizer = action.add_parser("synthesize", help="Run Graph Synthesizer")
    synthesizer.set_defaults(action=_run_synthesizer)
    synthesizer.add_argument(
        "-s",
        "--synthesizer",
        type=str,
        required=True,
        help=f"Synthesizer to use. Avaialble synthesizers: {list_available_synthesizer()}",
    )
    synthesizer.add_argument(
        "-dp", "--data-path", type=str, default=None, help="Path to dataset"
    )
    synthesizer.add_argument(
        "-pp",
        "--preprocessing",
        type=str,
        help=f"Preprocessing object to use, add custom preprocessing to\
                                     datasets available datasets: {list(DATASETS.keys())}",
    )
    synthesizer.add_argument(
        "-sp",
        "--save-path",
        type=str,
        default="./",
        required=False,
        help="Save path to dump generated files",
    )
    synthesizer.add_argument(
        "-a",
        "--aligner",
        type=str,
        default="",
        help=f"Aligner to use. Available aligners: {list_available_graph_aligner()}",
    )
    synthesizer.add_argument(
        "-gg",
        "--graph-generator",
        type=str,
        default=None,
        help=f"Graph generator to use to generate graph structure {list_available_graph_generators()} ",
    )
    synthesizer.add_argument("--gg-seed", type=int, default=None)
    synthesizer.add_argument(
        "-eg",
        "--edge-generator",
        type=str,
        default=None,
        help=f"Edge generator to use to generate edge features {list_available_tabular_generators()} ",
    )
    synthesizer.add_argument("--eg-batch-size", type=int, default=2000)
    synthesizer.add_argument("--eg-epochs", type=int, default=10)
    synthesizer.add_argument(
        "-ng",
        "--node-generator",
        type=str,
        default=None,
        help=f"Node generator to use to generate node features {list_available_tabular_generators()}",
    )
    synthesizer.add_argument("--ng-batch-size", type=int, default=2000)
    synthesizer.add_argument("--ng-epochs", type=int, default=10)
    synthesizer.add_argument("--num-workers", type=int, default=1)

    synthesizers = _get_synthesizers()
    for key in synthesizers:
        synthesizers[key].add_args(synthesizer)

    aligners = _get_aligners()
    for key in aligners.keys():
        aligners[key].add_args(synthesizer)

    tabular_generators = _get_tabular_generators()
    for key in tabular_generators.keys():
        tabular_generators[key].add_args(synthesizer)

    graph_generators = _get_graph_generators()
    for key in graph_generators.keys():
        graph_generators[key].add_args(synthesizer)

    # - pre-training
    pretraining = action.add_parser(
        "pretrain", help="Run Synthetic Graph Data Pre-training Tool"
    )
    pretraining.set_defaults(action=_run_pretraining)
    for key in MODELS.keys():
        MODELS[key].add_args(pretraining)

    paths = pretraining.add_argument_group("Paths")
    paths.add_argument(
        "--data-path",
        type=pathlib.Path,
        default=pathlib.Path("./data"),
        help=f"Directory where the data is located or should be downloaded",
    )

    optimizer = pretraining.add_argument_group("Optimizer")
    optimizer.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-3,
        help=f"Initial learning rate for optimizer",
    )
    optimizer.add_argument("--weight-decay", type=float, default=0.1)
    pretraining.add_argument(
        "-pp",
        "--preprocessing",
        type=str,
        help=f"Preprocessing object to use, add custom preprocessing to\
                                     datasets available datasets: {list(DATASETS.keys())}",
    )
    pretraining.add_argument(
        "--pretraining-path",
        type=str,
        default=None,
        help="Path to generated dataset",
    )
    pretraining.add_argument(
        "--model",
        type=str,
        default="gat_ec",
        help=f"List of available models: {list(MODELS.keys())}",
    )
    pretraining.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden feature dimension"
    )
    pretraining.add_argument(
        "--out-dim",
        type=int,
        default=32,
        help="Output feature dimension",
    )
    pretraining.add_argument(
        "--task",
        type=str,
        default="ec",
        help=f"now the only available option is ec (edge-classification)",
    )
    pretraining.add_argument(
        "--log-interval", type=int, default=5, help="logging interval"
    )
    pretraining.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Target column for downstream prediction",
    )
    pretraining.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of classes in the target column",
    )
    pretraining.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="Number of pre-training epochs",
    )
    pretraining.add_argument(
        "--finetune-epochs",
        type=int,
        default=0,
        help="Number of finetuning epochs",
    )
    pretraining.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Pre-training and Fine-tuning dataloader batch size",
    )
    pretraining.add_argument(
        "--seed", type=int, default=1341, help="Set a seed globally"
    )
    pretraining.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloading workers",
    )
    pretraining.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="Multi-layer full neighborhood sampler layers",
    )

    pretraining.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use as train",
    )
    pretraining.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use as val",
    )
    pretraining.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use as test",
    )
    pretraining.add_argument("--shuffle", action="store_true", default=False)
    pretraining.add_argument(
        "--silent",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Minimize stdout output",
    )
    pretraining.add_argument(
        "--timeit",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="Minimize stdout output",
    )

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()

    return args, sys.argv


def main():
    args, argv = get_args()
    log.info("=========================================")
    log.info("|    Synthetic Graph Generation Tool    |")
    log.info("=========================================")

    try:
        _ = args.action(args)
    except Exception as error:
        print(f"{error}")
        print(error.with_traceback())
    sys.exit(0)


if __name__ == "__main__":
    main()
