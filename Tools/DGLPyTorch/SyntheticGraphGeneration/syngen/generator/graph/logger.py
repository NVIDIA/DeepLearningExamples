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
import os
from datetime import datetime


class BaseLogger:
    """ Base logger class
    Args:
        logdir (str): path to the logging directory
    """

    def __init__(self, logdir: str = "tmp"):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        currentDateAndTime = datetime.now()
        self.logname = (
            f'{currentDateAndTime.strftime("%Y_%m_%d_%H_%M_%S")}.txt'
        )
        self.logpath = os.path.join(self.logdir, self.logname)
        self.setup_logger()
        self.log("Initialized logger")

    def setup_logger(self):
        """ This function setups logger """
        logging.basicConfig(
            filename=self.logpath,
            filemode="a",
            format="%(asctime)s| %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG,
        )

    def log(self, msg: str):
        """ This function logs messages in debug mode
        Args:
            msg (str): message to be printed
        """

        logging.debug(msg)
