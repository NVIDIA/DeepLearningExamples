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

import os
import sys

import inspect

from contextlib import contextmanager

from six import add_metaclass

import threading
import logging as _logging

import warnings

from mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from mask_rcnn.utils.metaclasses import SingletonMetaClass

__all__ = [
    "logging",
    "log_cleaning"
]

MODEL_NAME = "MaskRCNN"


class StdOutFormatter(_logging.Formatter):
    """
    Log formatter used in Tornado. Key features of this formatter are:
    * Color support when logging to a terminal that supports it.
    * Timestamps on every log line.
    * Robust against str/bytes encoding problems.
    """
    DEFAULT_FORMAT = '%(color)s[{model_name}] %(levelname)-8s: %(end_color)s%(message)s'.format(
        model_name=MODEL_NAME
    )

    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, fmt=None, datefmt=None, style='%'):
        r"""
        :arg bool color: Enables color support.
        :arg string fmt: Log message format.
          It will be applied to the attributes dict of log records. The
          text between ``%(color)s`` and ``%(end_color)s`` will be colored
          depending on the level if color support is on.
        :arg dict colors: color mappings from logging level to terminal color
          code
        :arg string datefmt: Datetime format.
          Used for formatting ``(asctime)`` placeholder in ``prefix_fmt``.
        .. versionchanged:: 3.2
           Added ``fmt`` and ``datefmt`` arguments.
        """

        if fmt is None:
            fmt = self.DEFAULT_FORMAT

        if datefmt is None:
            datefmt = self.DEFAULT_DATE_FORMAT

        # _logging.Formatter.__init__(self, datefmt=datefmt)
        super(StdOutFormatter, self).__init__(fmt=fmt, datefmt=datefmt, style=style)

        self._fmt = fmt
        self._colors = {}
        self._normal = ''

    def format(self, record):
        try:
            message = record.getMessage()
            assert isinstance(message, str)  # guaranteed by logging
            # Encoding notes:  The logging module prefers to work with character
            # strings, but only enforces that log messages are instances of
            # basestring.  In python 2, non-ascii bytestrings will make
            # their way through the logging framework until they blow up with
            # an unhelpful decoding error (with this formatter it happens
            # when we attach the prefix, but there are other opportunities for
            # exceptions further along in the framework).
            #
            # If a byte string makes it this far, convert it to unicode to
            # ensure it will make it out to the logs.  Use repr() as a fallback
            # to ensure that all byte strings can be converted successfully,
            # but don't do it by default so we don't add extra quotes to ascii
            # bytestrings.  This is a bit of a hacky place to do this, but
            # it's worth it since the encoding errors that would otherwise
            # result are so useless (and tornado is fond of using utf8-encoded
            # byte strings wherever possible).
            record.message = self.to_unicode(message)

        except Exception as e:
            record.message = "Bad message (%r): %r" % (e, record.__dict__)

        record.asctime = self.formatTime(record, self.datefmt)

        if record.levelno in self._colors:
            record.color = self._colors[record.levelno]
            record.end_color = self._normal
        else:
            record.color = record.end_color = ''

        formatted = self._fmt % record.__dict__

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            # exc_text contains multiple lines.  We need to _safe_unicode
            # each line separately so that non-utf8 bytes don't cause
            # all the newlines to turn into '\n'.
            lines = [formatted.rstrip()]
            lines.extend(self.to_unicode(ln) for ln in record.exc_text.split('\n'))

            formatted = '\n'.join(lines)
        return formatted.replace("\n", "\n    ")

    @staticmethod
    def to_unicode(value):
        """
        Converts a string argument to a unicode string.
        If the argument is already a unicode string or None, it is returned
        unchanged.  Otherwise it must be a byte string and is decoded as utf8.
        """
        try:
            if isinstance(value, (str, type(None))):
                return value

            if not isinstance(value, bytes):
                raise TypeError("Expected bytes, unicode, or None; got %r" % type(value))

            return value.decode("utf-8")

        except UnicodeDecodeError:
            return repr(value)



@add_metaclass(SingletonMetaClass)
class _Logger(object):

    # Level 0
    NOTSET = _logging.NOTSET

    # Level 10
    DEBUG = _logging.DEBUG

    # Level 20
    INFO = _logging.INFO

    # Level 30
    WARNING = _logging.WARNING

    # Level 40
    ERROR = _logging.ERROR

    # Level 50
    CRITICAL = _logging.CRITICAL

    _level_names = {
        0: 'NOTSET',
        10: 'DEBUG',
        20: 'INFO',
        30: 'WARNING',
        40: 'ERROR',
        50: 'CRITICAL',
    }

    def __init__(self, capture_io=True):

        self._logger = None
        self._logger_lock = threading.Lock()

        self._handlers = dict()

        self.old_warnings_showwarning = None

        if MPI_rank_and_size()[0] == 0:
            self._define_logger()

    def _define_logger(self):

        # Use double-checked locking to avoid taking lock unnecessarily.
        if self._logger is not None:
            return self._logger

        with self._logger_lock:

            try:
                # Scope the TensorFlow logger to not conflict with users' loggers.
                self._logger = _logging.getLogger(MODEL_NAME)
                self.reset_stream_handler()

            finally:
                self.set_verbosity(verbosity_level=_Logger.INFO)

            self._logger.propagate = False

    def reset_stream_handler(self):

        if self._logger is None:
            raise RuntimeError("Impossible to set handlers if the Logger is not predefined")

        # ======== Remove Handler if already existing ========

        try:
            self._logger.removeHandler(self._handlers["stream_stdout"])
        except KeyError:
            pass

        try:
            self._logger.removeHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

        # ================= Streaming Handler =================

        # Add the output handler.
        self._handlers["stream_stdout"] = _logging.StreamHandler(sys.stdout)
        self._handlers["stream_stdout"].addFilter(lambda record: record.levelno <= _logging.INFO)

        self._handlers["stream_stderr"] = _logging.StreamHandler(sys.stderr)
        self._handlers["stream_stderr"].addFilter(lambda record: record.levelno > _logging.INFO)

        Formatter = StdOutFormatter

        self._handlers["stream_stdout"].setFormatter(Formatter())
        self._logger.addHandler(self._handlers["stream_stdout"])

        try:
            self._handlers["stream_stderr"].setFormatter(Formatter())
            self._logger.addHandler(self._handlers["stream_stderr"])
        except KeyError:
            pass

    def get_verbosity(self):
        """Return how much logging output will be produced."""
        if self._logger is not None:
            return self._logger.getEffectiveLevel()

    def set_verbosity(self, verbosity_level):
        """Sets the threshold for what messages will be logged."""
        if self._logger is not None:
            self._logger.setLevel(verbosity_level)

            for handler in self._logger.handlers:
                handler.setLevel(verbosity_level)

    @contextmanager
    def temp_verbosity(self, verbosity_level):
        """Sets the a temporary threshold for what messages will be logged."""

        if self._logger is not None:

            old_verbosity = self.get_verbosity()

            try:
                self.set_verbosity(verbosity_level)
                yield

            finally:
                self.set_verbosity(old_verbosity)

        else:
            try:
                yield

            finally:
                pass

    def captureWarnings(self, capture):
        """
        If capture is true, redirect all warnings to the logging package.
        If capture is False, ensure that warnings are not redirected to logging
        but to their original destinations.
        """

        if self._logger is not None:

            if capture and self.old_warnings_showwarning is None:
                self.old_warnings_showwarning = warnings.showwarning  # Backup Method
                warnings.showwarning = self._showwarning

            elif not capture and self.old_warnings_showwarning is not None:
                warnings.showwarning = self.old_warnings_showwarning  # Restore Method
                self.old_warnings_showwarning = None

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        """
        Implementation of showwarnings which redirects to logging.
        It will call warnings.formatwarning and will log the resulting string
        with level logging.WARNING.
        """
        s = warnings.formatwarning(message, category, filename, lineno, line)
        self.warning("%s", s)

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(_Logger.DEBUG):
            self._logger._log(_Logger.DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(_Logger.INFO):
            self._logger._log(_Logger.INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(_Logger.WARNING):
            self._logger._log(_Logger.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(_Logger.ERROR):
            self._logger._log(_Logger.ERROR, msg, args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        if self._logger is not None and self._logger.isEnabledFor(_Logger.CRITICAL):
            self._logger._log(_Logger.CRITICAL, msg, args, **kwargs)


def log_cleaning(hide_deprecation_warnings=False):

    if hide_deprecation_warnings:
        warnings.simplefilter("ignore")

        from tensorflow.python.util import deprecation
        from tensorflow.python.util import deprecation_wrapper
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0

    formatter = _logging.Formatter('[%(levelname)s] %(message)s')

    from tensorflow.python.platform import tf_logging
    tf_logging.get_logger().propagate = False

    _logging.getLogger().propagate = False

    for handler in _logging.getLogger().handlers:
        handler.setFormatter(formatter)


# Necessary to catch the correct caller
_logging._srcfile = os.path.normcase(inspect.getfile(_Logger.__class__))


logging = _Logger()
