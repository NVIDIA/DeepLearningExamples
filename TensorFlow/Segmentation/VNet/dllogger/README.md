# DLLogger - minimal logging tool

This project emerged from the need for unified logging schema for Deep Learning Examples modules. It provides a simple, extensible and intuitive logging capabilities with API trimmed to absolute minimum.

## Installation
```bash
pip install dllogger
```

## Quick Start

To start using DLLogger you need to add just two lines of code and you are good to go!
```python
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
import dllogger as DLLogger

DLLogger.init(backends=[
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
])
```

To log anything you need to call `DLLogger.log(step=<TRAINING_STEP>, data=<DATA>, verbosity=<VERBOSITY>)`

`<TRAINING_STEP>` can be any number/string/tuple which would indicate where are we in the training process.
We propose a following convention:


- Use `step="PARAMETER"` for script parameters (everything that is needed to reproduce the result)

- Use a tuple of numbers to indicate training progress, for example:

  - `step=tuple(epoch_number, iteration_number, validation_iteration_number)` for a `validation_iteration_number` in a validation that happens after iteration `iteration_number` of epoch `epoch_number`

  - `step=tuple(epoch_number,)` for a summary of epoch `epoch_number`

  - `step=tuple()` for a summary of whole training run

`<DATA>` should be a dictionary with metric names as keys and metric values as values.

To log a metric metadata, for example unit, description, ordering, format call `DLLogger.metadata(metric_name, metric_metadata)` where metric metadata is a dictionary. Backends can use the metadata information
for logging purposes, for example StdOutBackend uses `format` and `unit` field to format its output.

Log is automatically saved on exit of python process (with exception of processes killed with SIGKILL) but if you want to flush log file before training ends:
```
DLLogger.flush()
```

Please refer to `examples/dllogger_example.py` and `examples/dllogger_singleton_example.py` files for example usage.

## Available backends overview

### StdOutBackend
Vanilla backend that holds no buffers. Just prints provided values to stdout.

```python
StdOutBackend(verbosity, step_format=..., metric_format=...)
```

-`step_format` is a function that formats step in DLLogger.log call

-`metric_format` is a function that formats a metric name and value given its metadata

For details see the `default_*_format` functions in `dllogger/logger.py`

Example output - see `examples/stdout.txt`

### JSONStreamBackend
```python
JsonBackend(verbosity, file_name)
```

JSONStreamBackend is saving JSON lines into a file. Example output - see `examples/dummy_resnet_log.json`

## Advanced usage

### Multiple Loggers
It is possible to to obtain Logger instance without referencing to DLLogger global instance.
```python
from dllogger import Logger
logger = Logger(backends=BACKEND_LIST)
```

