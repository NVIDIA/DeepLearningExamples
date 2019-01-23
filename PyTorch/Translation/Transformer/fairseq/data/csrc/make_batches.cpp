// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace at { namespace native {

namespace {
   bool is_batch_full(int64_t num_tokens, int64_t max_tokens, int64_t max_sentences, int64_t batch_length){
      if (batch_length == 0){
         return false;
      } else if (batch_length == max_sentences || num_tokens > max_tokens){
         return true;
      } else {
         return false;
      }
         
   }
}


std::vector<std::vector<int64_t> > make_batches(py::array_t<int64_t> src_lengths, py::array_t<int64_t> tgt_lengths, py::array_t<int64_t> idx_list, int64_t max_tokens, int64_t max_sentences, uint64_t bsz_mult, int64_t max_len){
   std::vector<std::vector<int64_t> > batches;   
   auto src_l = src_lengths.unchecked<1>();
   auto tgt_l = tgt_lengths.unchecked<1>();
   auto idx_l = idx_list.unchecked<1>();
   AT_ASSERTM(src_l.shape(0) == tgt_l.shape(0), "tgt_list and src_list should have the same shape");
   AT_ASSERTM(idx_l.shape(0) == tgt_l.shape(0), "idx_list and tgt_list should have the same shape");
   ssize_t nelem = src_l.shape(0);
   int64_t sample_len =0;
   std::vector<int64_t> sample_lens;
   std::vector<int64_t> batch; 
   for (ssize_t i=0; i < nelem; i++){
       int64_t idx = idx_l(i);
       int64_t sample_num_tokens = std::max(src_l(idx), tgt_l(idx));
       if (sample_num_tokens > max_len) continue;
       sample_len = std::max(sample_len, sample_num_tokens);
       sample_lens.push_back(sample_num_tokens);
       int64_t num_tokens = (batch.size() + 1) * sample_len;
       if (is_batch_full(num_tokens, max_tokens, max_sentences, batch.size())){
          int64_t mode_len = std::max(batch.size() / bsz_mult * bsz_mult, batch.size() % bsz_mult);
          std::vector<int64_t> new_batch;
          new_batch.reserve(mode_len);
          std::copy(batch.begin()+mode_len, batch.end(), std::back_inserter(new_batch)); 
          batch.erase(batch.begin()+mode_len, batch.end());
          sample_lens.erase(sample_lens.begin(), sample_lens.begin()+mode_len);
//sample_len always contains at least one element
          sample_len = *std::max_element(sample_lens.begin(), sample_lens.end());
          batches.push_back(batch);
          batch = new_batch;
       }
       batch.push_back(idx);
   }
   if (batch.size() > 0) batches.push_back(batch);
   return batches;
}   


}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("make_batches", &at::native::make_batches);
}
  
