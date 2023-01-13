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

from collections import Counter
import sys


in_ltr = sys.argv[1]
out_dict = sys.argv[2]

counter = Counter()
with open(in_ltr) as ltr:
    for line in ltr:
        counter.update(line[:-1].replace(" ", ""))

with open(out_dict, "w") as out:
    for letter, cnt in counter.most_common():
        out.write(f"{letter} {cnt}\n")
