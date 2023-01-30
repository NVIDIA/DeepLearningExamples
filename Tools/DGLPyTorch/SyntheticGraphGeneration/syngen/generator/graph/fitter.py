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

import warnings
from abc import ABC
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


class BaseFitter(ABC):
    """ Base fitter for graph generators """
    def fit(self, graph: Optional[List[Tuple[int, int]]], *args):
        raise NotImplementedError()


class RandomFitter(BaseFitter):
    """ Random fitter for RMAT generators. Implements erdos-renyi model. """
    def fit(self, graph: Optional[List[Tuple[int, int]]], *args):
        """ Fit method for erdos-renyi model, it is graph independent.
        Args:
            graph (List[Tuple[int, int]]): graph to be fitted on
        Returns:
            a, b, c, d: random seeding matrix
        """
        return 0.25, 0.25, 0.25, 0.25


MAXK = 1000

class FastFitter(BaseFitter):
    """ Fast version of fitter RMAT generators. Finds seeding matrix based on given degree distribution """
    def _convert_graph_to_fitformat(self, edges, in_out_degree):
        if in_out_degree == "out":
            combined = edges[:, 0]
        elif in_out_degree == "in":
            combined = edges[:, 1]
        else:
            raise ValueError(
                f"in_out_degree expected [out|in|both] got {in_out_degree}"
            )
        ids, counts = np.unique(combined, return_counts=True)
        node_dd = np.stack([ids, counts], axis=1)
        ids2, counts2 = np.unique(counts, return_counts=True)
        dd = np.stack([ids2, counts2], axis=1)
        num_nodes = len(ids)
        log2_num_nodes = int(np.ceil(np.log2(num_nodes)))
        return node_dd, dd, log2_num_nodes

    def _get_p_directed_graph(self, dd, n_exp2, verbose=False):
        E = (dd[:, 0] * dd[:, 1]).sum()

        mx = min(dd[-1, 0], MAXK)

        logeck = np.zeros(shape=(mx + 1), dtype=np.float64)
        tmp = 0
        for k in range(1, mx + 1):
            logeck[k] = tmp + np.log(E - k + 1) - np.log(k)
            tmp = logeck[k]

        lognci = np.zeros(shape=(n_exp2 + 1), dtype=np.float64)
        tmp = 0
        for i in range(1, n_exp2 + 1):
            lognci[i] = tmp + np.log(n_exp2 - i + 1) - np.log(i)
            tmp = lognci[i]

        x0 = np.array([0.5], dtype=np.float64)

        self.optimization_steps = []
        fun = lambda x: self._loglik(x, E, n_exp2, dd, logeck, lognci, MAXK)

        res = minimize(
            fun,
            x0,
            method="Nelder-Mead",
            bounds=[(1e-4, 1.0 - 1e-4)],
            options={"disp": verbose, "fatol": 1e-4},
        )
        return res.x[0]

    def _loglik(self, p, E, n_exp, count, logeck, lognci, k_cost_threeshold):

        if p <= 0.0 or p >= 1.0:
            return 1e100

        q = p
        a = 0.75 * p
        b = p - a
        c = q - a
        if (a + b + c) >= 1.0:
            return 1e100

        l = np.array([0.0], dtype=np.float64)
        for j in range(len(count)):
            m = count[j, 0]
            ck = np.log(count[j, 1])
            if ck > np.log(k_cost_threeshold):
                if m <= MAXK:
                    current_sum = np.exp(
                        logeck[m]
                        + np.log(p) * (n_exp * m)
                        + np.log(1 - p ** n_exp) * (E - m)
                    )

                    for i in range(1, n_exp + 1):
                        current_sum = current_sum + np.exp(
                            logeck[m]
                            + lognci[i]
                            + np.log(p) * (m * (n_exp - i))
                            + np.log(1 - p) * (m * i)
                            + np.log(1 - p ** (n_exp - i) * (1 - p) ** i)
                            * (E - m)
                        )
                else:
                    logecm = (
                        E * np.log(E) - m * np.log(m) - (E - m) * np.log(E - m)
                    )
                    current_sum = np.exp(
                        logecm
                        + np.log(p) * (n_exp * m)
                        + np.log(1 - p ** n_exp) * (E - m)
                    )
                    for i in range(1, n_exp + 1):
                        current_sum = current_sum + np.exp(
                            logecm
                            + lognci[i]
                            + np.log(p) * (m * (n_exp - i))
                            + np.log(1 - p) * (m * i)
                            + np.log(1 - p ** (n_exp - i) * (1 - p) ** i)
                            * (E - m)
                        )
                y = np.log(current_sum)
                y = max(0, y)
                l = l + (np.exp(ck) - np.exp(y)) * (np.exp(ck) - np.exp(y))
        self.optimization_steps.append((p[0], l[0]))
        return l

    def _check_optimization_history(self):
        optimization_steps = np.array(self.optimization_steps)
        function_values = np.unique(optimization_steps[:, 1])
        if len(function_values) <= 1:
            warnings.warn(
                "the optimization function is constant for the FastFitter. Please, use Fitter class instead."
            )
        self.optimization_steps = []

    def fit(self, graph: List[Tuple[int, int]], return_log_d=False):
        """ Fits the graph to the degree distribution.
        Args:
            graph (List[Tuple[int, int]]): graph to be fitted on
            return_log_d (bool): flag if set returns additionally log_2 from numbers of nodes
        Returns:
            a, b, c, d: seeding matrix
        """

        if not isinstance(graph, np.ndarray):
            graph = np.array(graph)

        (
            node_outdd,
            out_dd,
            log2_from_nnodes,
        ) = self._convert_graph_to_fitformat(graph, "out")
        node_indd, in_dd, log2_to_nnodes = self._convert_graph_to_fitformat(
            graph, "in"
        )

        p = self._get_p_directed_graph(out_dd, log2_from_nnodes)
        self._check_optimization_history()
        q = self._get_p_directed_graph(in_dd, log2_to_nnodes)
        self._check_optimization_history()

        a = 0.75 * (p + q) / 2
        b = p - a
        c = q - a
        assert (a + b + c) < 1.0, "Cannot get correct RMat fit!"
        d = 1.0 - (a + b + c)
        if return_log_d:
            return log2_from_nnodes, log2_to_nnodes, a, b, c, d
        return a, b, c, d


class Fitter(FastFitter):
    """ Full version of fitter RMAT generators. Finds seeding matrix based on given degree distribution """

    def _loglik(self, p, E, n_exp, count, logeck, lognci, k_cost_threeshold):

        if p <= 0.0 or p >= 1.0:
            return 1e100

        q = p
        a = 0.75 * p
        b = p - a
        c = q - a
        if (a + b + c) >= 1.0:
            return 1e100

        Sx = 0.0
        Sx2 = 0.0
        Sx3 = 0.0
        Sx4 = 0.0
        Sy = 0.0
        Sxy = 0.0
        Sx2y = 0.0

        numX = count[-1, 0]

        totObs = 0.0
        prevY = 0.0

        for m in range(1, numX + 1):
            x = np.log(m)
            if m <= MAXK:
                current_sum = np.exp(
                    logeck[m]
                    + np.log(p) * (n_exp * m)
                    + np.log(1 - p ** n_exp) * (E - m)
                )

                for i in range(1, n_exp + 1):
                    current_sum = current_sum + np.exp(
                        logeck[m]
                        + lognci[i]
                        + np.log(p) * (m * (n_exp - i))
                        + np.log(1.0 - p) * (m * i)
                        + np.log(1.0 - p ** (n_exp - i) * (1.0 - p) ** i)
                        * (E - m)
                    )
            else:
                logecm = (
                    E * np.log(E) - m * np.log(m) - (E - m) * np.log(E - m)
                )
                current_sum = np.exp(
                    logecm
                    + np.log(p) * (n_exp * m)
                    + np.log(1 - p ** n_exp) * (E - m)
                )
                for i in range(1, n_exp + 1):
                    current_sum = current_sum + np.exp(
                        logecm
                        + lognci[i]
                        + np.log(p) * (m * (n_exp - i))
                        + np.log(1.0 - p) * (m * i)
                        + np.log(1.0 - p ** (n_exp - i) * (1.0 - p) ** i)
                        * (E - m)
                    )

            y = np.log(current_sum)
            y = max(0, y)

            interpY = y

            while interpY > 0 and (m == 1 or x > np.log(m - 1)):
                Sx = Sx + x
                Sx2 = Sx2 + x * x
                Sx3 = Sx3 + x * x * x
                Sx4 = Sx4 + x * x * x * x
                Sy = Sy + interpY
                Sxy = Sxy + x * interpY
                Sx2y = Sx2y + x * x * interpY

                x = x - (np.log(numX) - np.log(numX - 1))

                if prevY <= 0:
                    interpY = 0
                else:
                    interpY = interpY - (interpY - prevY) / (
                        np.log(m) - np.log(m - 1)
                    ) * (np.log(numX) - np.log(numX - 1))

                totObs = totObs + 1

            prevY = y

        res = np.linalg.inv(
            np.array([[totObs, Sx, Sx2], [Sx, Sx2, Sx3], [Sx2, Sx3, Sx4]])
        ) @ np.array([Sy, Sxy, Sx2y])

        ParabolaA = res[0]
        ParabolaB = res[1]
        ParabolaC = res[2]

        l = np.array([0.0], dtype=np.float64)

        for m in range(1, len(count) + 1):
            k = np.log(count[m - 1, 1])
            expectedLogY = (
                ParabolaA
                + ParabolaB * np.log(count[m - 1, 0])
                + ParabolaC * np.log(count[m - 1, 0]) * np.log(count[m - 1, 0])
            )
            l = l + (k - expectedLogY) * (k - expectedLogY)

        self.optimization_steps.append((p[0], l[0]))
        return l
