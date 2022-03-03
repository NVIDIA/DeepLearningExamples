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

import numpy as np
import torch
from run_squad import convert_examples_to_features, read_squad_examples
from tokenization import BertTokenizer


def get_dataloader_fn(
    precision : str = 'fp32',
    batch_size: int = 8,
    vocab_file: str = "",
    do_lower_case: bool = True,
    predict_file: str = "",
    max_len: int = 512,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    max_query_length: int = 64,
    version_2_with_negative: bool = False,
    pad_to_batch_size: bool = True, 
):

    # Preprocess input data
    tokenizer = BertTokenizer(vocab_file, do_lower_case=do_lower_case, max_len=max_len)

    eval_examples = read_squad_examples(
        input_file=predict_file, is_training=False, version_2_with_negative=version_2_with_negative
    )
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
    )

    # get inputs 
    all_unique_ids = [f.unique_id for f in eval_features]
    all_input_ids = [f.input_ids for f in eval_features]
    all_input_mask = [f.input_mask for f in eval_features]
    all_segment_ids = [f.segment_ids for f in eval_features]

    if pad_to_batch_size:
        # each batch should have a fixed size 
        f = eval_features[-1]
        padding = batch_size - (len(all_unique_ids) % batch_size)
        all_unique_ids += [f.unique_id for _ in range(padding)]
        all_input_ids += [f.input_ids for _ in range(padding)]
        all_input_mask += [f.input_mask for _ in range(padding)]
        all_segment_ids += [f.segment_ids for _ in range(padding)]

    all_unique_ids = torch.tensor(all_unique_ids, dtype=torch.int32, requires_grad=False)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.int32, requires_grad=False)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.int32, requires_grad=False)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.int32, requires_grad=False)
    eval_data = torch.utils.data.TensorDataset(all_unique_ids, all_input_ids, all_input_mask, all_segment_ids)
    eval_sampler = torch.utils.data.SequentialSampler(eval_data)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    dtype = { 'fp32' : np.float32, 'fp16' : np.float16 }
    dtype = dtype[precision]
    
    def _get_dataloader():
        """return dataloader for inference"""
        for unique_id, input_ids, input_mask, segment_ids in eval_dataloader:
            unique_id = unique_id.cpu().numpy()
            input_ids = input_ids.cpu().numpy()
            input_mask = input_mask.cpu().numpy()
            segment_ids = segment_ids.cpu().numpy()
            x = {"input__0": input_ids, "input__1": segment_ids, "input__2": input_mask}
            y_real = {
                "output__0": np.zeros([batch_size, max_seq_length], dtype=dtype),
                "output__1": np.zeros([batch_size, max_seq_length], dtype=dtype),
            }
            yield (unique_id, x, y_real)

    return _get_dataloader

