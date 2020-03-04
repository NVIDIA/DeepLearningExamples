// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
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

#pragma once

#define HAVE_CUDA 1  // Loading Kaldi headers with GPU

#include <cfloat>
#include <sstream>
#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

using kaldi::BaseFloat;

namespace nvidia {
namespace inferenceserver {
namespace custom {
namespace kaldi_cbe {

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(const std::string& instance_name, const ModelConfig& config,
          const int gpu_device);
  virtual ~Context();

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(const uint32_t payload_cnt, CustomPayload* payloads,
              CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  // init kaldi pipeline
  int InitializeKaldiPipeline();
  int InputOutputSanityCheck();
  int ReadModelParameters();
  int GetSequenceInput(CustomGetNextInputFn_t& input_fn, void* input_context,
                       CorrelationID* corr_id, int32_t* start, int32_t* ready,
                       int32_t* dim, int32_t* end,
                       const kaldi::BaseFloat** wave_buffer,
                       std::vector<uint8_t>* input_buffer);

  int SetOutputTensor(const std::string& output, CustomGetOutputFn_t output_fn,
                      CustomPayload payload);

  bool CheckPayloadError(const CustomPayload& payload);
  int FlushBatch();

  // The name of this instance of the backend.
  const std::string instance_name_;

  // The model configuration.
  const ModelConfig model_config_;

  // The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  // execute on CPU.
  const int gpu_device_;

  // Models paths
  std::string nnet3_rxfilename_, fst_rxfilename_;
  std::string word_syms_rxfilename_;

  // batch_size
  int max_batch_size_;
  int num_channels_;
  int num_worker_threads_;
  std::vector<CorrelationID> batch_corr_ids_;
  std::vector<kaldi::SubVector<kaldi::BaseFloat>> batch_wave_samples_;
  std::vector<bool> batch_is_last_chunk_;

  BaseFloat sample_freq_, seconds_per_chunk_;
  int chunk_num_bytes_, chunk_num_samps_;

  // feature_config includes configuration for the iVector adaptation,
  // as well as the basic features.
  kaldi::cuda_decoder::BatchedThreadedNnet3CudaOnlinePipelineConfig
      batched_decoder_config_;
  std::unique_ptr<kaldi::cuda_decoder::BatchedThreadedNnet3CudaOnlinePipeline>
      cuda_pipeline_;
  // Maintain the state of some shared objects
  kaldi::TransitionModel trans_model_;

  kaldi::nnet3::AmNnetSimple am_nnet_;
  fst::SymbolTable* word_syms_;

  const uint64_t int32_byte_size_;
  const uint64_t int64_byte_size_;
  std::vector<int64_t> output_shape_;

  std::vector<uint8_t> byte_buffer_;
  std::vector<std::vector<uint8_t>> wave_byte_buffers_;
};

}  // kaldi
}  // custom
}  // inferenceserver
}  // nvidia
