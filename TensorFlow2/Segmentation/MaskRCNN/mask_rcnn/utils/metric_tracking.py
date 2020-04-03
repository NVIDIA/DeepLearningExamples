#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import weakref

from mask_rcnn.utils.logging_backend import DistributedStrategy
from mask_rcnn.utils.logging_backend import LoggingScope

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils import meters

__all__ = ["TF_METRICS", "KERAS_MODELS", "KERAS_OPTIMIZERS", "register_metric", "clear_registered_metrics"]


class WeakRefList(object):
    def __init__(self):
        self._items = list()

    def _clean_iternal_list(self):
        self._items = [s for s in self._items if s() is not None]

    def __iter__(self):
        self._clean_iternal_list()

        for obj in self._items:
            if obj() is None:
                continue

            yield obj()

    def __len__(self):
        self._clean_iternal_list()
        return len(self._items)

    def clear(self):
        self._items.clear()

    def append(self, new_item):
        self._items.append(weakref.ref(new_item))
        self._clean_iternal_list()


TF_METRICS = dict()
KERAS_MODELS = WeakRefList()
KERAS_OPTIMIZERS = WeakRefList()


def register_metric(
    name,
    tensor,
    aggregator=meters.StandardMeter(),
    metric_scope=LoggingScope.ITER,
    distributed_strategy=DistributedStrategy.NONE
):

    if name in TF_METRICS.keys():
        raise ValueError("A metric with the name `%s` has already been registered" % name)

    if not issubclass(aggregator.__class__, meters.AbstractMeterMixin):
        raise ValueError("Unknown `aggregator` received: %s" % aggregator.__class__.__name__)

    if metric_scope not in LoggingScope.__values__():
        raise ValueError(
            "Unknown `metric_scope` received: %s, authorized: %s" %
            (metric_scope, LoggingScope.__values__())
        )

    if distributed_strategy not in DistributedStrategy.__values__():
        raise ValueError(
            "Unknown `distributed_strategy` received: %s, authorized: %s" %
            (distributed_strategy, DistributedStrategy.__values__())
        )

    TF_METRICS[name] = {
        "tensor": tensor,
        "aggregator": aggregator,
        "distributed_strategy": distributed_strategy,
        "scope": metric_scope,
    }

    logging.debug(
        "New Metric Registered: `{metric_name}`, Aggregator: {aggregator}, "
        "Scope: {scope}, Distributed Strategy: {distributed_strategy}".format(
            metric_name=name, aggregator=str(aggregator), distributed_strategy=distributed_strategy, scope=metric_scope
        )
    )


def clear_registered_metrics():
    TF_METRICS.clear()
    logging.debug("All registered metrics have been cleared")
