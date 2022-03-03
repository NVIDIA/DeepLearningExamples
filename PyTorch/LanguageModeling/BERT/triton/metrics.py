# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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


import json
import logging
import os
import subprocess
import sys
from argparse import Namespace
from typing import Any, Dict, List, Optional

import numpy as np
from run_squad import RawResult, convert_examples_to_features, get_answers, read_squad_examples
from tokenization import BertTokenizer

# 
from triton.deployment_toolkit.core import BaseMetricsCalculator


class MetricsCalculator(BaseMetricsCalculator):
    def __init__(
        self,
        eval_script: str = "data/squad/v1.1/evaluate-v1.1.py",
        predict_file: str = "",
        output_dir: str = "./",
        n_best_size: int = 20,
        max_answer_length: int = 30,
        version_2_with_negative: bool = False,
        max_seq_length: int = 384,
        doc_stride: int = 128,
        max_query_length: int = 64,
        vocab_file: str = "",
        do_lower_case: bool = True,
        max_len: int = 512,
    ):

        tokenizer = BertTokenizer(vocab_file, do_lower_case=do_lower_case, max_len=max_len)  # for bert large

        self.eval_examples = read_squad_examples(
            input_file=predict_file, is_training=False, version_2_with_negative=version_2_with_negative
        )

        self.eval_features = convert_examples_to_features(
            examples=self.eval_examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
        )

        self.output_dir = output_dir
        self.eval_script = eval_script
        self.predict_file = predict_file

        args = Namespace(
            version_2_with_negative=version_2_with_negative,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            verbose_logging=False,
            do_lower_case=do_lower_case,
        )

        self.args = args

        self.all_results: List[RawResult] = []

    def _calc(self) -> Dict[str, float]:
        dataset_size = len(self.eval_features)
        self.all_results = self.all_results[:dataset_size]
        output_prediction_file = os.path.join(self.output_dir, "predictions.json")
        answers, _ = get_answers(self.eval_examples, self.eval_features, self.all_results, self.args)
        with open(output_prediction_file, "w") as f:
            f.write(json.dumps(answers, indent=4) + "\n")

        eval_out = subprocess.check_output(
            [sys.executable, self.eval_script, self.predict_file, output_prediction_file]
        )
        scores = str(eval_out).strip()
        # exact_match = float(scores.split(":")[1].split(",")[0])
        f1 = float(scores.split(":")[2].split("}")[0])

        return {"f1": f1}

    def update(
        self,
        ids: List[Any],
        y_pred: Dict[str, np.ndarray],
        x: Optional[Dict[str, np.ndarray]],
        y_real: Optional[Dict[str, np.ndarray]],
    ):
        start_logits = y_pred["output__0"]
        end_logits = y_pred["output__1"]

        for unique_id, start_logit, end_logit in zip(ids, start_logits, end_logits):
            start_logit = start_logit.tolist()
            end_logit = end_logit.tolist()
            raw_result = RawResult(unique_id=unique_id, start_logits=start_logit, end_logits=end_logit)
            self.all_results.append(raw_result)

    @property
    def metrics(self) -> Dict[str, float]:
        return self._calc()

