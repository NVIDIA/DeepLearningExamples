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

"""Utilities that interact with cloud service.
"""

import requests

GCP_METADATA_URL = "http://metadata/computeMetadata/v1/instance/hostname"
GCP_METADATA_HEADER = {"Metadata-Flavor": "Google"}


def on_gcp():
  """Detect whether the current running environment is on GCP."""
  try:
    # Timeout in 5 seconds, in case the test environment has connectivity issue.
    # There is not default timeout, which means it might block forever.
    response = requests.get(
        GCP_METADATA_URL, headers=GCP_METADATA_HEADER, timeout=5)
    return response.status_code == 200
  except requests.exceptions.RequestException:
    return False
