# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import itertools

import numpy as np
from pytablewriter import MarkdownTableWriter


def interleave(*args):
    return list(itertools.chain(*zip(*args)))


class AccuracyTable:
    def __init__(self, unit):
        self.data = collections.defaultdict(dict)
        self.unit = unit

    def add(self, key, data):
        self.data[key].update(data)

    def write(self, title, write_math):
        writer = MarkdownTableWriter()
        writer.table_name = f'{title}'
        main_header = ['**Batch Size**', '**Beam Size**']
        data_header = []
        if 'fp32' in write_math:
            data_header += [f'**Accuracy - FP32 ({self.unit})**']
        if 'fp16' in write_math:
            data_header += [f'**Accuracy - FP16 ({self.unit})**']
        writer.headers = main_header + data_header

        writer.value_matrix = []
        for k, v in self.data.items():
            batch_size, beam_size = k
            row = [batch_size, beam_size]
            if 'fp32' in write_math:
                row.append(v['fp32'])
            if 'fp16' in write_math:
                row.append(v['fp16'])
            writer.value_matrix.append(row)
        writer.write_table()


class PerformanceTable:
    def __init__(self, percentiles, unit, reverse_percentiles=False):
        self.percentiles = percentiles
        self.data = collections.defaultdict(dict)
        self.unit = unit
        self.reverse_percentiles = reverse_percentiles

    def add(self, key, value):
        math, value = next(iter(value.items()))
        value = np.array(value)

        if self.reverse_percentiles:
            percentiles = [100 - p for p in self.percentiles]
        else:
            percentiles = self.percentiles

        stats = []
        for p in percentiles:
            val = np.percentile(value, p)
            stats.append(val * self.unit_convert[self.unit])

        avg = value.mean() * self.unit_convert[self.unit]

        self.data[key].update({math: (avg, stats)})

    def write(self, title, math, relative=None, reverse_speedup=False):
        writer = MarkdownTableWriter()
        writer.table_name = f'{title} - {math.upper()}'
        main_header = ['**Batch Size**', '**Beam Size**']
        data_header = [f'**Avg ({self.unit})**']
        data_header += [f'**{p}% ({self.unit})**' for p in self.percentiles]

        if relative:
            speedup_header = ['**Speedup**'] * len(data_header)
            data_header = interleave(data_header, speedup_header)

        writer.headers = main_header + data_header

        writer.value_matrix = []
        for k, v in self.data.items():
            batch_size, beam_size = k
            avg, res_percentiles = v[math]
            main = [batch_size, beam_size]
            data = [avg, *res_percentiles]

            if relative:
                rel = self.data[k][relative]
                rel_avg, rel_res_percentiles = rel
                rel = [rel_avg, *rel_res_percentiles]
                speedup = [d / r for (r, d) in zip(rel, data)]
                if reverse_speedup:
                    speedup = [1 / s for s in speedup]
                data = interleave(data, speedup)

            writer.value_matrix.append(main + data)
        writer.write_table()


class LatencyTable(PerformanceTable):
    def __init__(self, percentiles, unit='ms'):
        super().__init__(percentiles, unit)
        self.unit_convert = {'s': 1, 'ms': 1e3, 'us': 1e6}


class ThroughputTable(PerformanceTable):
    def __init__(self, percentiles, unit='tok/s', reverse_percentiles=True):
        super().__init__(percentiles, unit, reverse_percentiles)
        self.unit_convert = {'tok/s': 1}
