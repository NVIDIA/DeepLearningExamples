# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import tensorflow as tf
from sacrebleu import corpus_bleu

def bleu_score(pred_file, ref_file):
    with tf.io.gfile.GFile(pred_file) as pred_stream, tf.io.gfile.GFile(ref_file) as ref_stream:
        bleu = corpus_bleu(pred_stream, [ref_stream], force=True)
        print("       bleu score: {:6.2f}".format(bleu.score))
        print("       bleu counts: {}".format(bleu.counts))
        print("       bleu totals: {}".format(bleu.totals))
        print("       bleu precisions: {}".format(bleu.precisions))
        print("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
        return bleu
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("[ERROR] bleu_score.py needs a result file and a solution file. \n e.g. python bleu_score.py f1.txt f2.txt")
        sys.exit(0)
    bleu_score(sys.argv[1], sys.argv[2])
