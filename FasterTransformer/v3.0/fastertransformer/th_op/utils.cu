/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastertransformer/th_op/utils.h"

namespace torch_ext {

// modified from TensorFlow's implementation of tf.contrib.seq2seq.gather_tree
__global__ void gather_tree_kernel(const int batch_size, const int max_time, const int beam_width, const int end_token,
                                   const int* step_ids, const int* parent_ids, const int* max_sequence_lengths, int* beams) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size * beam_width; i += gridDim.x * blockDim.x) {
    const int batch = i / beam_width;
    const int beam = i % beam_width;

    const int max_seq_len_b = min(max_time, __ldg(max_sequence_lengths + batch));
    if (max_seq_len_b <= 0) {
      continue;
    }

#define GET_IX(time_ix, beam_ix) (batch_size * beam_width * (time_ix) + beam_width * batch + (beam_ix))

    const int initial_beam_ix = GET_IX(max_seq_len_b - 1, beam);
    beams[initial_beam_ix] = __ldg(step_ids + initial_beam_ix);
    int parent = __ldg(parent_ids + initial_beam_ix);
    bool found_bad = false;
    for (int level = max_seq_len_b - 2; level >= 0; --level) {
      const int level_beam_ix = GET_IX(level, beam);
      const int level_parent_ix = GET_IX(level, parent);
      if (parent < 0 || parent > beam_width) {
        beams[level_beam_ix] = -1;
        parent = -1;
        found_bad = true;
      } else {
        beams[level_beam_ix] = __ldg(step_ids + level_parent_ix);
        parent = __ldg(parent_ids + level_parent_ix);
      }
    }
// Not necessary when using a BeamSearchDecoder, but necessary
// when a user feeds in possibly broken trajectory (i.e., non-eos
// entries in a beam following eos entries).
    if (!found_bad) {
      bool finished = false;
      for (int time = 0; time < max_seq_len_b; ++time) {
        const int level_beam_ix = GET_IX(time, beam);
        if (finished) {
          beams[level_beam_ix] = end_token;
        } else if (beams[level_beam_ix] == end_token) {
          finished = true;
        }
      }
    }
#undef GET_IX
  }
}


void gather_tree_kernel_launcher(int max_time, int batch_size, int beam_width,
                                 int* step_ids, int* parent_ids, int* max_sequence_lengths,
                                 int end_token, int* beams, cudaStream_t stream) {
  int batchbeam = batch_size * beam_width;
  dim3 grid(1), block(batchbeam);
  // though decoder do not support > 1024 for now
  if (batchbeam > 1024) {
    grid.x = ceil(batch_size * beam_width / 1024.);
    block.x = 1024;
  }
  gather_tree_kernel<<<grid, block, 0, stream>>>(batch_size, max_time, beam_width, end_token,
                                                 step_ids, parent_ids, max_sequence_lengths, beams);
}
} // namespace torch_ext
