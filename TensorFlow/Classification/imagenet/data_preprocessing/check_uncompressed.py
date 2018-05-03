#!/usr/bin/python
# Copyright 2017 NVIDIA Corporation. All Rights Reserved.
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
import glob
import re
import sys
import tarfile

if __name__ == '__main__':
  tar_file_name = sys.argv[1]
  path_to_check = sys.argv[2]
  tar_file = tarfile.open(tar_file_name)
  tar_paths = set([_i.path for _i in tar_file])
  uncompressed_paths = files = [re.sub('(\.?\/)?(.+)', '\g<2>', _f) for _f in glob.glob(path_to_check + "/*")]
  for _f in tar_paths:
    if _f not in uncompressed_paths:
      print("%s is not fully uncompressed" % tar_file_name)
      print("Missing file: %s in %s" % (_f, path_to_check))
      sys.exit(1)
  print("%s is already fully uncompressed" % tar_file_name)

