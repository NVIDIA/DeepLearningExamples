# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Logging utilities for benchmark.

For collecting local environment metrics like CPU and memory, certain python
packages need be installed. See README for details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import datetime
import json
import multiprocessing
import numbers
import os
import threading
import uuid

from six.moves import _thread as thread
from absl import flags
import tensorflow as tf
from tensorflow.python.client import device_lib

from official.utils.logs import cloud_lib

METRIC_LOG_FILE_NAME = "metric.log"
BENCHMARK_RUN_LOG_FILE_NAME = "benchmark_run.log"
_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"
GCP_TEST_ENV = "GCP"
RUN_STATUS_SUCCESS = "success"
RUN_STATUS_FAILURE = "failure"
RUN_STATUS_RUNNING = "running"


FLAGS = flags.FLAGS

# Don't use it directly. Use get_benchmark_logger to access a logger.
_benchmark_logger = None
_logger_lock = threading.Lock()


def config_benchmark_logger(flag_obj=None):
  """Config the global benchmark logger."""
  _logger_lock.acquire()
  try:
    global _benchmark_logger
    if not flag_obj:
      flag_obj = FLAGS

    if (not hasattr(flag_obj, "benchmark_logger_type") or
        flag_obj.benchmark_logger_type == "BaseBenchmarkLogger"):
      _benchmark_logger = BaseBenchmarkLogger()
    elif flag_obj.benchmark_logger_type == "BenchmarkFileLogger":
      _benchmark_logger = BenchmarkFileLogger(flag_obj.benchmark_log_dir)
    elif flag_obj.benchmark_logger_type == "BenchmarkBigQueryLogger":
      from official.benchmark import benchmark_uploader as bu  # pylint: disable=g-import-not-at-top
      bq_uploader = bu.BigQueryUploader(gcp_project=flag_obj.gcp_project)
      _benchmark_logger = BenchmarkBigQueryLogger(
          bigquery_uploader=bq_uploader,
          bigquery_data_set=flag_obj.bigquery_data_set,
          bigquery_run_table=flag_obj.bigquery_run_table,
          bigquery_run_status_table=flag_obj.bigquery_run_status_table,
          bigquery_metric_table=flag_obj.bigquery_metric_table,
          run_id=str(uuid.uuid4()))
    else:
      raise ValueError("Unrecognized benchmark_logger_type: %s"
                       % flag_obj.benchmark_logger_type)

  finally:
    _logger_lock.release()
  return _benchmark_logger


def get_benchmark_logger():
  if not _benchmark_logger:
    config_benchmark_logger()
  return _benchmark_logger


@contextlib.contextmanager
def benchmark_context(flag_obj):
  """Context of benchmark, which will update status of the run accordingly."""
  benchmark_logger = config_benchmark_logger(flag_obj)
  try:
    yield
    benchmark_logger.on_finish(RUN_STATUS_SUCCESS)
  except Exception:  # pylint: disable=broad-except
    # Catch all the exception, update the run status to be failure, and re-raise
    benchmark_logger.on_finish(RUN_STATUS_FAILURE)
    raise


class BaseBenchmarkLogger(object):
  """Class to log the benchmark information to STDOUT."""

  def log_evaluation_result(self, eval_results):
    """Log the evaluation result.

    The evaluate result is a dictionary that contains metrics defined in
    model_fn. It also contains a entry for global_step which contains the value
    of the global step when evaluation was performed.

    Args:
      eval_results: dict, the result of evaluate.
    """
    if not isinstance(eval_results, dict):
      tf.compat.v1.logging.warning(
          "eval_results should be dictionary for logging. Got %s",
          type(eval_results))
      return
    global_step = eval_results[tf.compat.v1.GraphKeys.GLOBAL_STEP]
    for key in sorted(eval_results):
      if key != tf.compat.v1.GraphKeys.GLOBAL_STEP:
        self.log_metric(key, eval_results[key], global_step=global_step)

  def log_metric(self, name, value, unit=None, global_step=None, extras=None):
    """Log the benchmark metric information to local file.

    Currently the logging is done in a synchronized way. This should be updated
    to log asynchronously.

    Args:
      name: string, the name of the metric to log.
      value: number, the value of the metric. The value will not be logged if it
        is not a number type.
      unit: string, the unit of the metric, E.g "image per second".
      global_step: int, the global_step when the metric is logged.
      extras: map of string:string, the extra information about the metric.
    """
    metric = _process_metric_to_json(name, value, unit, global_step, extras)
    if metric:
      tf.compat.v1.logging.info("Benchmark metric: %s", metric)

  def log_run_info(self, model_name, dataset_name, run_params, test_id=None):
    tf.compat.v1.logging.info(
        "Benchmark run: %s", _gather_run_info(model_name, dataset_name,
                                              run_params, test_id))

  def on_finish(self, status):
    pass


class BenchmarkFileLogger(BaseBenchmarkLogger):
  """Class to log the benchmark information to local disk."""

  def __init__(self, logging_dir):
    super(BenchmarkFileLogger, self).__init__()
    self._logging_dir = logging_dir
    if not tf.io.gfile.isdir(self._logging_dir):
      tf.io.gfile.makedirs(self._logging_dir)
    self._metric_file_handler = tf.io.gfile.GFile(
        os.path.join(self._logging_dir, METRIC_LOG_FILE_NAME), "a")

  def log_metric(self, name, value, unit=None, global_step=None, extras=None):
    """Log the benchmark metric information to local file.

    Currently the logging is done in a synchronized way. This should be updated
    to log asynchronously.

    Args:
      name: string, the name of the metric to log.
      value: number, the value of the metric. The value will not be logged if it
        is not a number type.
      unit: string, the unit of the metric, E.g "image per second".
      global_step: int, the global_step when the metric is logged.
      extras: map of string:string, the extra information about the metric.
    """
    metric = _process_metric_to_json(name, value, unit, global_step, extras)
    if metric:
      try:
        json.dump(metric, self._metric_file_handler)
        self._metric_file_handler.write("\n")
        self._metric_file_handler.flush()
      except (TypeError, ValueError) as e:
        tf.compat.v1.logging.warning(
            "Failed to dump metric to log file: name %s, value %s, error %s",
            name, value, e)

  def log_run_info(self, model_name, dataset_name, run_params, test_id=None):
    """Collect most of the TF runtime information for the local env.

    The schema of the run info follows official/benchmark/datastore/schema.

    Args:
      model_name: string, the name of the model.
      dataset_name: string, the name of dataset for training and evaluation.
      run_params: dict, the dictionary of parameters for the run, it could
        include hyperparameters or other params that are important for the run.
      test_id: string, the unique name of the test run by the combination of key
        parameters, eg batch size, num of GPU. It is hardware independent.
    """
    run_info = _gather_run_info(model_name, dataset_name, run_params, test_id)

    with tf.io.gfile.GFile(os.path.join(
        self._logging_dir, BENCHMARK_RUN_LOG_FILE_NAME), "w") as f:
      try:
        json.dump(run_info, f)
        f.write("\n")
      except (TypeError, ValueError) as e:
        tf.compat.v1.logging.warning(
            "Failed to dump benchmark run info to log file: %s", e)

  def on_finish(self, status):
    self._metric_file_handler.flush()
    self._metric_file_handler.close()


class BenchmarkBigQueryLogger(BaseBenchmarkLogger):
  """Class to log the benchmark information to BigQuery data store."""

  def __init__(self,
               bigquery_uploader,
               bigquery_data_set,
               bigquery_run_table,
               bigquery_run_status_table,
               bigquery_metric_table,
               run_id):
    super(BenchmarkBigQueryLogger, self).__init__()
    self._bigquery_uploader = bigquery_uploader
    self._bigquery_data_set = bigquery_data_set
    self._bigquery_run_table = bigquery_run_table
    self._bigquery_run_status_table = bigquery_run_status_table
    self._bigquery_metric_table = bigquery_metric_table
    self._run_id = run_id

  def log_metric(self, name, value, unit=None, global_step=None, extras=None):
    """Log the benchmark metric information to bigquery.

    Args:
      name: string, the name of the metric to log.
      value: number, the value of the metric. The value will not be logged if it
        is not a number type.
      unit: string, the unit of the metric, E.g "image per second".
      global_step: int, the global_step when the metric is logged.
      extras: map of string:string, the extra information about the metric.
    """
    metric = _process_metric_to_json(name, value, unit, global_step, extras)
    if metric:
      # Starting new thread for bigquery upload in case it might take long time
      # and impact the benchmark and performance measurement. Starting a new
      # thread might have potential performance impact for model that run on
      # CPU.
      thread.start_new_thread(
          self._bigquery_uploader.upload_benchmark_metric_json,
          (self._bigquery_data_set,
           self._bigquery_metric_table,
           self._run_id,
           [metric]))

  def log_run_info(self, model_name, dataset_name, run_params, test_id=None):
    """Collect most of the TF runtime information for the local env.

    The schema of the run info follows official/benchmark/datastore/schema.

    Args:
      model_name: string, the name of the model.
      dataset_name: string, the name of dataset for training and evaluation.
      run_params: dict, the dictionary of parameters for the run, it could
        include hyperparameters or other params that are important for the run.
      test_id: string, the unique name of the test run by the combination of key
        parameters, eg batch size, num of GPU. It is hardware independent.
    """
    run_info = _gather_run_info(model_name, dataset_name, run_params, test_id)
    # Starting new thread for bigquery upload in case it might take long time
    # and impact the benchmark and performance measurement. Starting a new
    # thread might have potential performance impact for model that run on CPU.
    thread.start_new_thread(
        self._bigquery_uploader.upload_benchmark_run_json,
        (self._bigquery_data_set,
         self._bigquery_run_table,
         self._run_id,
         run_info))
    thread.start_new_thread(
        self._bigquery_uploader.insert_run_status,
        (self._bigquery_data_set,
         self._bigquery_run_status_table,
         self._run_id,
         RUN_STATUS_RUNNING))

  def on_finish(self, status):
    self._bigquery_uploader.update_run_status(
        self._bigquery_data_set,
        self._bigquery_run_status_table,
        self._run_id,
        status)


def _gather_run_info(model_name, dataset_name, run_params, test_id):
  """Collect the benchmark run information for the local environment."""
  run_info = {
      "model_name": model_name,
      "dataset": {"name": dataset_name},
      "machine_config": {},
      "test_id": test_id,
      "run_date": datetime.datetime.utcnow().strftime(
          _DATE_TIME_FORMAT_PATTERN)}
  _collect_tensorflow_info(run_info)
  _collect_tensorflow_environment_variables(run_info)
  _collect_run_params(run_info, run_params)
  _collect_cpu_info(run_info)
  _collect_memory_info(run_info)
  _collect_test_environment(run_info)
  return run_info


def _process_metric_to_json(
    name, value, unit=None, global_step=None, extras=None):
  """Validate the metric data and generate JSON for insert."""
  if not isinstance(value, numbers.Number):
    tf.compat.v1.logging.warning(
        "Metric value to log should be a number. Got %s", type(value))
    return None

  extras = _convert_to_json_dict(extras)
  return {
      "name": name,
      "value": float(value),
      "unit": unit,
      "global_step": global_step,
      "timestamp": datetime.datetime.utcnow().strftime(
          _DATE_TIME_FORMAT_PATTERN),
      "extras": extras}


def _collect_tensorflow_info(run_info):
  run_info["tensorflow_version"] = {
      "version": tf.version.VERSION, "git_hash": tf.version.GIT_VERSION}


def _collect_run_params(run_info, run_params):
  """Log the parameter information for the benchmark run."""
  def process_param(name, value):
    type_check = {
        str: {"name": name, "string_value": value},
        int: {"name": name, "long_value": value},
        bool: {"name": name, "bool_value": str(value)},
        float: {"name": name, "float_value": value},
    }
    return type_check.get(type(value),
                          {"name": name, "string_value": str(value)})
  if run_params:
    run_info["run_parameters"] = [
        process_param(k, v) for k, v in sorted(run_params.items())]


def _collect_tensorflow_environment_variables(run_info):
  run_info["tensorflow_environment_variables"] = [
      {"name": k, "value": v}
      for k, v in sorted(os.environ.items()) if k.startswith("TF_")]


# The following code is mirrored from tensorflow/tools/test/system_info_lib
# which is not exposed for import.
def _collect_cpu_info(run_info):
  """Collect the CPU information for the local environment."""
  cpu_info = {}

  cpu_info["num_cores"] = multiprocessing.cpu_count()

  try:
    # Note: cpuinfo is not installed in the TensorFlow OSS tree.
    # It is installable via pip.
    import cpuinfo    # pylint: disable=g-import-not-at-top

    info = cpuinfo.get_cpu_info()
    cpu_info["cpu_info"] = info["brand"]
    cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6

    run_info["machine_config"]["cpu_info"] = cpu_info
  except ImportError:
    tf.compat.v1.logging.warn(
        "'cpuinfo' not imported. CPU info will not be logged.")


def _collect_memory_info(run_info):
  try:
    # Note: psutil is not installed in the TensorFlow OSS tree.
    # It is installable via pip.
    import psutil   # pylint: disable=g-import-not-at-top
    vmem = psutil.virtual_memory()
    run_info["machine_config"]["memory_total"] = vmem.total
    run_info["machine_config"]["memory_available"] = vmem.available
  except ImportError:
    tf.compat.v1.logging.warn(
        "'psutil' not imported. Memory info will not be logged.")


def _collect_test_environment(run_info):
  """Detect the local environment, eg GCE, AWS or DGX, etc."""
  if cloud_lib.on_gcp():
    run_info["test_environment"] = GCP_TEST_ENV
  # TODO(scottzhu): Add more testing env detection for other platform


def _parse_gpu_model(physical_device_desc):
  # Assume all the GPU connected are same model
  for kv in physical_device_desc.split(","):
    k, _, v = kv.partition(":")
    if k.strip() == "name":
      return v.strip()
  return None


def _convert_to_json_dict(input_dict):
  if input_dict:
    return [{"name": k, "value": v} for k, v in sorted(input_dict.items())]
  else:
    return []
