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

import abc
import json
import pickle
from abc import ABC
from typing import List, Optional, Set, Tuple

import numpy as np

from syngen.generator.graph.utils import BaseLogger
from syngen.generator.graph.seeder import BaseSeeder


class BaseGenerator(abc.ABC):
    """ BaseGenerator class """

    JSON_ASSERTION = "Expected file to be json"

    @classmethod
    def get_generators(cls, include_parents=True):
        """ Recursively find subclasses
        Args:
            include_parents (bool): whether to include parents to other classes. (default: `True`)
        Returns:
            generators: dictionary with all the subclasses
        """

        generators = dict()
        for child in cls.__subclasses__():
            children = child.get_generators(include_parents)
            generators.update(children)

            if include_parents or not children:
                if abc.ABC not in child.__bases__ and BaseGenerator not in child.__bases__:
                    generators[child.__name__] = child
        return generators

    def save(self, path):
        raise NotImplementedError()

    @classmethod
    def load(cls, path):
        raise NotImplementedError()


class BaseGraphGenerator(BaseGenerator, ABC):
    """ Base class for all graph generators
    Args:
        *args: optional positional args
        **kwargs: optional key-word args
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        logdir: str = "./logs",
        gpu: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self._fit_results = None
        self.seeder = BaseSeeder(seed)
        self.seeder.reseed()
        self.logger = BaseLogger(logdir)
        self.logger.log(f"Using seed: {self.seeder.seed}")
        self.gpu = gpu
        self.verbose = verbose

    def fit(
        self, graph: List[Tuple[int, int]], is_directed: bool, *args, **kwargs
    ):
        """ Fits generator on the graph
        Args: 
            graph (List[Tuple[int, int]]): graph to be fitted on
            is_directed (bool): flag indicating whether the graph is directed
            *args: optional positional args
            **kwargs: optional key-word args
        """
        raise NotImplementedError()

    def generate(
        self,
        num_nodes: int,
        num_edges: int,
        is_directed: bool,
        *args,
        return_node_ids: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """ Generates graph with approximately `num_nodes` and exactly `num_edges` from generator
        Args: 
            num_nodes (int): approximate number of nodes to be generated
            num_edges (int): exact number of edges to be generated
            is_directed (bool): flag indicating whether the generated graph has to be directed
            return_node_ids (bool): flag indicating whether the generator has to return nodes_ids as the second output
            *args: optional positional args
            **kwargs: optional key-word args
        """
        raise NotImplementedError()

    def set_fit_results(self, fit_results):
        self._fit_results = fit_results

    def get_fit_results(self):
        return self._fit_results

    def save_fit_results(self, save_path: str = "./fit_results.json"):
        """ Store fitted results into json file
        Args:
            save_path (str): path to the json file with the fitted result
        """
        assert (
            self._fit_results
        ), "There are no fit results to be saved, \
        call fit method first or load the results from the file"
        assert save_path.endswith(".json"), self.JSON_ASSERTION
        with open(save_path, "w") as fjson:
            json.dump(self._fit_results, fjson)

    def load_fit_results(self, load_path: str = "./fit_results.json"):
        """load fitted results from json file
        Args:  
            load_path (str): path to the json file with the fitted result
        """
        assert load_path.endswith(".json"), self.JSON_ASSERTION
        with open(load_path, "r") as fjson:
            self._fit_results = json.load(fjson)

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model

    @staticmethod
    def add_args(parser):
        return parser


class BaseBipartiteGraphGenerator(BaseGenerator, ABC):
    """ Base class for all bipartite graph generators
    Args:
        *args: optional positional args
        **kwargs: optional key-word args
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        logdir: str = "./logs",
        gpu: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self._fit_src_dst_results = None
        self._fit_dst_src_results = None
        self.seeder = BaseSeeder(seed)
        self.seeder.reseed()
        self.logger = BaseLogger(logdir)
        self.logger.log(f"Using seed: {self.seeder.seed}")
        self.gpu = gpu
        self.verbose = verbose

    def fit(
        self,
        graph: List[Tuple[int, int]],
        src_set: Set[int],
        dst_set: Set[int],
        is_directed: bool,
        transform_graph: bool,
        *args,
        **kwargs,
    ):
        """ Fits generator on the graph

        Args:
            graph (List[Tuple[int, int]]): graph to be fitted on
            src_set (Set[int]): set of source nodes
            dst_set (Set[int]): set of destination nodes
            is_directed (bool): flag indicating whether the graph is directed
            *args: optional positional args
            **kwargs: optional key-word args
        """
        raise NotImplementedError()

    def generate(
        self,
        num_nodes_src_set: int,
        num_nodes_dst_set: int,
        num_edges_src_dst: int,
        num_edges_dst_src: int,
        is_directed: bool,
        return_node_ids: bool = False,
        transform_graph: bool = True,
        *args,
        **kwargs,
    ):
        """ Generates graph with approximately `num_nodes_src_set`/`num_nodes_dst_set` nodes
         and exactly `num_edges_src_dst`/`num_edges_dst_src` edges from generator
        Args:
            num_nodes_src_set (int): approximate number of source nodes to be generated
            num_nodes_dst_set (int): approximate number of destination nodes to be generated
            num_edges_src_dst (int): exact number of source->destination edges to be generated
            num_edges_dst_src (int): exact number of destination->source to be generated
            is_directed (bool) flag indicating whether the generated graph has to be directed
            return_node_ids (bool): flag indicating whether the generator has to return nodes_ids as the second output
            *args: optional positional args
            **kwargs: optional key-word args
        """
        raise NotImplementedError()

    def set_fit_results(self, fit_results):
        self._fit_src_dst_results, self._fit_dst_src_results = fit_results

    def get_fit_results(self):
        return self._fit_src_dst_results, self._fit_dst_src_results

    def save_fit_results(self, save_path: str = "./fit_results.json"):
        """ Stores fitted results into json file
        Args:
            save_path (str): path to the json file with the fitted result
        """
        assert (
            self._fit_src_dst_results or self._fit_dst_src_results
        ), "There are no fit results to be saved, \
        call fit method first or load the results from the file"
        assert save_path.endswith(".json"), self.JSON_ASSERTION

        wrapped_results = {
            "fit_src_dst_results": self._fit_src_dst_results,
            "fit_dst_src_results": self._fit_dst_src_results,
        }

        with open(save_path, "w") as fjson:
            json.dump(wrapped_results, fjson)

    def load_fit_results(self, load_path: str = "./fit_results.json"):
        """ Loads fitted results from json file
        Args:
            load_path (str): path to the json file with the fitted result
        """
        assert load_path.endswith(".json"), self.JSON_ASSERTION
        with open(load_path, "r") as fjson:
            wrapped_results = json.load(fjson)
        assert (
            "fit_src_dst_results" in wrapped_results
            and "fit_dst_src_results" in wrapped_results
        ), "Required keys fit_src_dst_results and fit_dst_src_results keys in the json not found"
        self._fit_src_dst_results = wrapped_results["fit_src_dst_results"]
        self._fit_dst_src_results = wrapped_results["fit_dst_src_results"]

    def save(self, path):
        with open(path, 'wb') as file_handler:
            pickle.dump(self, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_handler:
            model = pickle.load(file_handler)
        return model

    @staticmethod
    def add_args(parser):
        return parser
