# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
from __future__ import print_function

import collections
import json
import re
import sys
from collections import defaultdict

from logger import tags
import logger.logger as nvl


LogLine = collections.namedtuple('LogLine', [
    'full_string',  # the complete line as a string
    'worker',       # the worker id
    'token',        # the token, i.e. ':::NVLOG'
    'version_str',  # the version string, e.g. 'v0.1.0'
    'model',        # the model, e.g. 'ncf'
    'timestamp',    # seconds as a float, e.g. 1234.567
    'filename',     # the which generated the log line, e.g. './convert.py'
    'lineno',       # the line in the file which generated the log line, e.g. 119
    'tag',          # the string tag
    'value',        # the parsed value associated with the tag, or None if no value
    'epoch',        # the epoch number of -1 if none
    'iteration',    # the interation number of -1 if none
    'scope'         # run, epoch, iteration, eval_iteration
])


def get_dict_value(x):
    if isinstance(x, dict):
        return x
    return {"value": x}


def get_value(x):
    if isinstance(x, dict):
        if "value" in x:
            return x.get("value")
        else:
            return x
    return x


def get_named_value(x, name):
    if isinstance(x, dict):
        if name in x:
            return x.get(name)
        else:
            return None
    return x


class NVLogParser(object):

    def __init__(self, token=nvl.NVLOGGER_TOKEN, version=nvl.NVLOGGER_VERSION):

        self.epoch = defaultdict(lambda: -1)
        self.iteration = defaultdict(lambda: -1)
        self.scope = defaultdict(lambda: 0)

        self.version = version
        self.token = token
        self.line_pattern = (
            '^'
            '([\d]?)'                    # optional worker id (0)
            '(' + token + ')'            # mandatory token (1)
            'v([\d]+\.[\d+]\.[\d+])[ ]'  # mandatory version (2)
            '([A-Za-z0-9_]+)[ ]'         # mandatory model (3)
            '([\d\.]+)[ ]'               # mandatory timestamp (4)
            '\(([^: ]+)'                 # mandatory file (5)
            ':(\d+)\)[ ]'                # mandatory lineno (6)
            '([A-Za-z0-9_]+)[ ]?'        # mandatory tag (7)
            '(:\s+(.+))?'                # optional value (8)
            '$'
        )
        # print(self.line_pattern)

        self.line_regex = re.compile(self.line_pattern, re.X)

    def string_to_logline(self, string):

        m = self.line_regex.match(string)

        if m is None:
            raise ValueError('does not match regex')

        args = [
            m.group(0),  # full string
        ]

        # by default
        worker = m.group(1)
        if worker == "":
            worker = "(0)"

        args.append(worker)

        args.append(m.group(2))  # token
        args.append(m.group(3))  # version
        args.append(m.group(4))  # model

        try:
            ts = float(m.group(5))      # parse timestamp
            args.append(ts)
        except ValueError:
            raise ValueError('timestamp format incorrect')

        args.append(m.group(6))         # file name

        try:
            lineno = int(m.group(7))    # may raise error
            args.append(lineno)
        except ValueError:
            raise ValueError('line number format incorrect')

        tag = m.group(8)
        args.append(tag)         # tag

        # 9th is ignored

        value = m.group(10)

        if value is not None:
            j = json.loads(value)
            args.append(j)
        else:
            # no Value
            args.append(None)

        # update processing state
        if tag == tags.TRAIN_EPOCH_START or tag == tags.TRAIN_EPOCH:
            self.epoch[worker] = get_named_value(value, tags.VALUE_EPOCH)
            self.scope[worker] = nvl.EPOCH_SCOPE
            self.iteration[worker] = -1

        if tag == tags.TRAIN_EPOCH_STOP:
            self.scope[worker] = nvl.RUN_SCOPE

        if tag == tags.TRAIN_ITER_START:
            self.iteration[worker] = get_named_value(value, tags.VALUE_ITERATION)
            self.scope[worker] = nvl.TRAIN_ITER_SCOPE

        if tag == tags.TRAIN_ITER_STOP:
            self.scope[worker] = nvl.EPOCH_SCOPE

        if tag == tags.PERF_IT_PER_SEC:
            self.scope[worker] = nvl.EPOCH_SCOPE

        if tag == tags.PERF_TIME_TO_TRAIN:
            self.scope[worker] = nvl.RUN_SCOPE

        args.append(self.epoch[worker])
        args.append(self.iteration[worker])
        args.append(self.scope[worker])

        return LogLine(*args)

    def parse_generator(self, gen):
        worker_loglines = defaultdict(list)
        loglines = []
        failed = []

        # state init for parsing
        self.epoch.clear()
        self.iteration.clear()
        self.scope.clear()

        for line in gen:
            line = line.strip()
            if line.find(self.token) == -1:
                continue
            try:
                ll = self.string_to_logline(line)
                worker_loglines[ll.worker].append(ll)
                loglines.append(ll)
            except ValueError as e:
                failed.append((line, str(e)))

        return loglines, failed, worker_loglines

    def parse_file(self, filename):
        with open(filename) as f:
            return self.parse_generator(f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: parser.py FILENAME')
        print('       tests parsing on the file.')
        sys.exit(1)

    filename = sys.argv[1]
    parser = NVLogParser()
    loglines, errors, worker_loglines = parser.parse_file(filename)

    print('Parsed {} log lines with {} errors.'.format(len(loglines), len(errors)))
    print('Found workers: {}.'.format(list(worker_loglines.keys())))

    if len(errors) > 0:
        print('Lines which failed to parse:')
        for line, error in errors:
            print('  Following line failed: {}'.format(error))
            print(line)

    if len(loglines) > 0:
        print('Lines which where parsed sucessfully:')
        for line in loglines:
            print(line.full_string, " ---> ", line.epoch, line.iteration, line.scope)

