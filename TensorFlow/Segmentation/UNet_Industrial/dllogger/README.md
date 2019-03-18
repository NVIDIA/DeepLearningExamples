# Tools for logging DL training
DLLogger is a tool to generate logs during Deep Learning training.

## Installation
```
git clone https://gitlab-master.nvidia.com/dl/JoC/DLLogger.git
pip install DLLogger/.
```

## Usage
You can use DLLogger with the simplest `LOGGER.log()` API:
```
from logger.logger import LOGGER
from logger import tags

LOGGER.model = 'ResNet'
LOGGER.log(key=tags.INPUT_BATCH_SIZE, value=128)
```
For the more advanced usage, please refer to the `dummy_run.py` example.

## Tags
All available tags are listed in the `logger/tags.py` file.
