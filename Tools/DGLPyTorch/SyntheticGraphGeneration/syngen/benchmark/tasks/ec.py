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

import logging
from time import perf_counter, time

import dgl
import numpy as np
import torch
from syngen.benchmark.data_loader.datasets.edge_ds import EdgeDS
from syngen.benchmark.models import MODELS
from syngen.utils.types import MetaData
from syngen.configuration import SynGenDatasetFeatureSpec

logger = logging.getLogger(__name__)
log = logger

_NAME = "edge classification"


def train_ec(
    args,
    finetune_feature_spec: SynGenDatasetFeatureSpec,
    *,
    pretrain_feature_spec: SynGenDatasetFeatureSpec = None,
):
    """Example edge classification training loop to pre-train on generated dataset
       with option to further finetune on a `finetune_source` dataset.
    """
    model = MODELS[args.model]
    optimizer = None
    out = {}
    dataset = EdgeDS(**vars(args))

    # - pre-training
    if pretrain_feature_spec is not None:
        # - dataset
        g, edge_ids = dataset.get_graph(
            pretrain_feature_spec, args.pretraining_edge_name
        )
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
        dataloader = dgl.dataloading.EdgeDataLoader(
            g,
            edge_ids,
            sampler,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            drop_last=False,
            num_workers=args.num_workers,
        )
        # - Model
        in_feats = g.ndata.get("feat").shape[1]
        in_feats_edge = g.edata.get("feat").shape[1]
        model = model(in_dim=in_feats, in_dim_edge=in_feats_edge, **vars(args))
        model = model.cuda()
        # - Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        log.info("Running pretraining ...")
        losses, times = [], []
        best_val_acc, best_test_acc = 0, 0
        # - Training loop
        for e in range(args.pretrain_epochs):

            if args.timeit:
                t0 = time.time()
            train_acc, val_acc, test_acc, losses = train_epoch(
                model, dataloader, optimizer
            )
            if args.timeit:
                t1 = time.time()
                times.append(t1 - t0)

            val_acc = np.mean(val_acc)
            test_acc = np.mean(test_acc)
            train_acc = np.mean(train_acc)
            loss = np.mean(losses)

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            if e % args.log_interval == 0:
                log.info(
                    "Pretraining epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                        e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                    )
                )

        out = {
            "pretrain-loss": loss,
            "pretrain-val-acc": val_acc,
            "pretrain-test-acc": test_acc,
            **out,
        }

        if args.timeit:
            out["pretrain-epoch-times"] = times


    g, edge_ids = dataset.get_graph(
        finetune_feature_spec, args.edge_name,
    )

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)

    dataloader = dgl.dataloading.EdgeDataLoader(
        g,
        edge_ids,
        sampler,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if optimizer is None:
        in_feats = g.ndata.get("feat").shape[1]
        in_feats_edge = g.edata.get("feat").shape[1]
        model = model(
            in_dim=in_feats, in_dim_edge=in_feats_edge, **vars(args)
        )

        model = model.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # - finetune
    best_val_acc, best_test_acc = 0, 0
    for e in range(args.finetune_epochs):
        if args.timeit:
            t0 = time.time()
        train_acc, val_acc, test_acc, losses = train_epoch(
            model, dataloader, optimizer
        )
        if args.timeit:
            t1 = time.time()
            times.append(t1 - t0)

        val_acc = np.mean(val_acc)
        test_acc = np.mean(test_acc)
        train_acc = np.mean(train_acc)
        loss = np.mean(losses)

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if e % args.log_interval == 0:
            log.info(
                "Finetuning: In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                    e, loss, val_acc, best_val_acc, test_acc, best_test_acc
                )
            )

    out = {
        "finetune-loss": loss,
        "finetune-val-acc": val_acc,
        "finetune-test-acc": test_acc,
        **out,
    }

    if args.timeit:
        out["finetune-epoch-times"] = times

    return out


def train_epoch(model, dataloader, optimizer, verbose=False):
    train_acc = []
    val_acc = []
    test_acc = []
    losses = []
    if verbose:
        times = []
        epoch_start = perf_counter()
    for input_nodes, edge_subgraph, blocks in dataloader:
        blocks = [b.to(torch.device("cuda")) for b in blocks]
        edge_subgraph = edge_subgraph.to(torch.device("cuda"))
        input_features = blocks[0].srcdata["feat"]
        edge_labels = edge_subgraph.edata["labels"]
        edge_features = None
        if "feat" in edge_subgraph.edata:
            edge_features = edge_subgraph.edata["feat"]
        edge_predictions = model(
            blocks=blocks,
            edge_subgraph=edge_subgraph,
            input_features=input_features,
            edge_features=edge_features,
        )
        train_mask = edge_subgraph.edata["train_mask"]
        val_mask = edge_subgraph.edata["val_mask"]
        test_mask = edge_subgraph.edata["test_mask"]
        loss = model.loss(
            edge_predictions[train_mask],
            torch.nn.functional.one_hot(
                edge_labels[train_mask].long(),
                num_classes=edge_predictions.shape[-1],
            ).float(),
        )
        # - store results
        losses.append(loss.item())
        preds = edge_predictions.argmax(1)
        train_acc.append(
            (preds[train_mask] == edge_labels[train_mask]).float().mean().item()
        )
        val_acc.append(
            (preds[val_mask] == edge_labels[val_mask]).float().mean().item()
        )
        test_acc.append(
            (preds[test_mask] == edge_labels[test_mask]).float().mean().item()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            times.append(perf_counter() - epoch_start)
            epoch_start = perf_counter()
    if verbose:
        return train_acc, val_acc, test_acc, losses, times
    return train_acc, val_acc, test_acc, losses
