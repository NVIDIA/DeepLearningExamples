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

from pytablewriter import MarkdownTableWriter


class TrainingTable:
    def __init__(self, acc_unit='BLEU', time_unit='min', perf_unit='tok/s'):
        self.data = []
        self.acc_unit = acc_unit
        self.time_unit = time_unit
        self.perf_unit = perf_unit
        self.time_unit_convert = {'s': 1, 'min': 1/60, 'h': 1/3600}

    def add(self, gpus, batch_size, accuracy, perf, time_to_train):
        time_to_train *= self.time_unit_convert[self.time_unit]
        if not accuracy:
            accuracy = 0.0
        accuracy = round(accuracy, 2)
        self.data.append([gpus, batch_size, accuracy, perf, time_to_train])

    def write(self, title, math):
        writer = MarkdownTableWriter()
        writer.table_name = f'{title}'

        header = [f'**GPUs**',
                  f'**Batch Size / GPU**',
                  f'**Accuracy - {math.upper()} ({self.acc_unit})**',
                  f'**Throughput - {math.upper()} ({self.perf_unit})**',
                  f'**Time to Train - {math.upper()} ({self.time_unit})**',
                  ]
        writer.headers = header

        writer.value_matrix = self.data
        writer.write_table()
