# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

DATASET_DIR=/data/wmt14_en_de_joined_dict
TEXT=examples/translation/wmt14_en_de

(
  cd examples/translation
  bash prepare-wmt14en2de.sh --scaling18
)

python preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir ${DATASET_DIR} \
  --nwordssrc 33712 \
  --nwordstgt 33712 \
  --joined-dictionary

cp $TEXT/code $DATASET_DIR/code
cp $TEXT/tmp/valid.raw.de $DATASET_DIR/valid.raw.de
sacrebleu -t wmt14/full -l en-de --echo ref > $DATASET_DIR/test.raw.de
