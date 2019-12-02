# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
import json
import atexit


class Backend(ABC):
    def __init__(self, verbosity):
        self._verbosity = verbosity

    @property
    def verbosity(self):
        return self._verbosity

    @abstractmethod
    def log(self, timestamp, elapsedtime, step, data):
        pass

    @abstractmethod
    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass


class Verbosity:
    OFF = -1
    DEFAULT = 0
    VERBOSE = 1


class Logger:
    def __init__(self, backends):
        self.backends = backends
        atexit.register(self.flush)
        self.starttime = datetime.now()

    def metadata(self, metric, metadata):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            b.metadata(timestamp, elapsedtime, metric, metadata)

    def log(self, step, data, verbosity=1):
        timestamp = datetime.now()
        elapsedtime = (timestamp - self.starttime).total_seconds()
        for b in self.backends:
            if b.verbosity >= verbosity:
                b.log(timestamp, elapsedtime, step, data)

    def flush(self):
        for b in self.backends:
            b.flush()


def default_step_format(step):
    return str(step)


def default_metric_format(metric, metadata, value):
    unit = metadata["unit"] if "unit" in metadata.keys() else ""
    format = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    return "{} : {} {}".format(
        metric, format.format(value) if value is not None else value, unit
    )


def default_prefix_format(timestamp):
    return "DLL {} - ".format(timestamp)


class StdOutBackend(Backend):
    def __init__(
        self,
        verbosity,
        step_format=default_step_format,
        metric_format=default_metric_format,
        prefix_format=default_prefix_format,
    ):
        super().__init__(verbosity=verbosity)

        self._metadata = defaultdict(dict)
        self.step_format = step_format
        self.metric_format = metric_format
        self.prefix_format = prefix_format

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        self._metadata[metric].update(metadata)

    def log(self, timestamp, elapsedtime, step, data):
        print(
            "{}{} {}".format(
                self.prefix_format(timestamp),
                self.step_format(step),
                " ".join(
                    [
                        self.metric_format(m, self._metadata[m], v)
                        for m, v in data.items()
                    ]
                ),
            )
        )

    def flush(self):
        pass


class JSONStreamBackend(Backend):
    def __init__(self, verbosity, filename):
        super().__init__(verbosity=verbosity)
        self._filename = filename
        self.file = open(filename, "w")
        atexit.register(self.file.close)

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        self.file.write(
            "DLLL {}\n".format(
                json.dumps(
                    dict(
                        timestamp=str(timestamp.timestamp()),
                        elapsedtime=str(elapsedtime),
                        datetime=str(timestamp),
                        type="METADATA",
                        metric=metric,
                        metadata=metadata,
                    )
                )
            )
        )

    def log(self, timestamp, elapsedtime, step, data):
        self.file.write(
            "DLLL {}\n".format(
                json.dumps(
                    dict(
                        timestamp=str(timestamp.timestamp()),
                        datetime=str(timestamp),
                        elapsedtime=str(elapsedtime),
                        type="LOG",
                        step=step,
                        data=data,
                    )
                )
            )
        )

    def flush(self):
        self.file.flush()
