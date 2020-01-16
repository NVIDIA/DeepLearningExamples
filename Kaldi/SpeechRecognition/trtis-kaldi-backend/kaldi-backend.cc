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

#include "kaldi-backend.h"
#include "kaldi-backend-utils.h"

namespace nvidia {
namespace inferenceserver {
namespace custom {
namespace kaldi_cbe {

Context::Context(const std::string& instance_name,
                 const ModelConfig& model_config, const int gpu_device)
    : instance_name_(instance_name),
      model_config_(model_config),
      gpu_device_(gpu_device),
      num_channels_(
          model_config_
              .max_batch_size()),  // diff in def between kaldi and trtis
      int32_byte_size_(GetDataTypeByteSize(TYPE_INT32)),
      int64_byte_size_(GetDataTypeByteSize(TYPE_INT64)) {}

Context::~Context() { delete word_syms_; }

int Context::ReadModelParameters() {
  // Reading config
  float beam, lattice_beam;
  int max_active;
  int frame_subsampling_factor;
  float acoustic_scale;
  int num_worker_threads;
  int err =
      ReadParameter(model_config_, "mfcc_filename",
                    &batched_decoder_config_.feature_opts.mfcc_config) ||
      ReadParameter(
          model_config_, "ivector_filename",
          &batched_decoder_config_.feature_opts.ivector_extraction_config) ||
      ReadParameter(model_config_, "beam", &beam) ||
      ReadParameter(model_config_, "lattice_beam", &lattice_beam) ||
      ReadParameter(model_config_, "max_active", &max_active) ||
      ReadParameter(model_config_, "frame_subsampling_factor",
                    &frame_subsampling_factor) ||
      ReadParameter(model_config_, "acoustic_scale", &acoustic_scale) ||
      ReadParameter(model_config_, "nnet3_rxfilename", &nnet3_rxfilename_) ||
      ReadParameter(model_config_, "fst_rxfilename", &fst_rxfilename_) ||
      ReadParameter(model_config_, "word_syms_rxfilename",
                    &word_syms_rxfilename_) ||
      ReadParameter(model_config_, "num_worker_threads", &num_worker_threads) ||
      ReadParameter(model_config_, "max_execution_batch_size",
                    &max_batch_size_);
  if (err) return err;
  max_batch_size_ = std::max<int>(max_batch_size_, 1);
  num_channels_ = std::max<int>(num_channels_, 1);

  // Sanity checks
  if (beam <= 0) return kInvalidModelConfig;
  if (lattice_beam <= 0) return kInvalidModelConfig;
  if (max_active <= 0) return kInvalidModelConfig;
  if (acoustic_scale <= 0) return kInvalidModelConfig;
  if (num_worker_threads <= 0) return kInvalidModelConfig;
  if (num_channels_ <= max_batch_size_) return kInvalidModelConfig;

  batched_decoder_config_.compute_opts.frame_subsampling_factor =
      frame_subsampling_factor;
  batched_decoder_config_.compute_opts.acoustic_scale = acoustic_scale;
  batched_decoder_config_.decoder_opts.default_beam = beam;
  batched_decoder_config_.decoder_opts.lattice_beam = lattice_beam;
  batched_decoder_config_.decoder_opts.max_active = max_active;
  batched_decoder_config_.num_worker_threads = num_worker_threads;
  batched_decoder_config_.max_batch_size = max_batch_size_;
  batched_decoder_config_.num_channels = num_channels_;

  auto feature_config = batched_decoder_config_.feature_opts;
  kaldi::OnlineNnet2FeaturePipelineInfo feature_info(feature_config);
  sample_freq_ = feature_info.mfcc_opts.frame_opts.samp_freq;
  BaseFloat frame_shift = feature_info.FrameShiftInSeconds();
  seconds_per_chunk_ = chunk_num_samps_ / sample_freq_;

  int samp_per_frame = static_cast<int>(sample_freq_ * frame_shift);
  float n_input_framesf = chunk_num_samps_ / samp_per_frame;
  bool is_integer = (n_input_framesf == std::floor(n_input_framesf));
  if (!is_integer) {
    std::cerr << "WAVE_DATA dim must be a multiple fo samples per frame ("
              << samp_per_frame << ")" << std::endl;
    return kInvalidModelConfig;
  }
  int n_input_frames = static_cast<int>(std::floor(n_input_framesf));
  batched_decoder_config_.compute_opts.frames_per_chunk = n_input_frames;

  return kSuccess;
}

int Context::InitializeKaldiPipeline() {
  batch_corr_ids_.reserve(max_batch_size_);
  batch_wave_samples_.reserve(max_batch_size_);
  batch_is_last_chunk_.reserve(max_batch_size_);
  wave_byte_buffers_.resize(max_batch_size_);
  output_shape_ = {1, 1};
  kaldi::CuDevice::Instantiate()
      .SelectAndInitializeGpuIdWithExistingCudaContext(gpu_device_);
  kaldi::CuDevice::Instantiate().AllowMultithreading();

  // Loading models
  {
    bool binary;
    kaldi::Input ki(nnet3_rxfilename_, &binary);
    trans_model_.Read(ki.Stream(), binary);
    am_nnet_.Read(ki.Stream(), binary);

    kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
    kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
    kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(),
                                &(am_nnet_.GetNnet()));
  }
  fst::Fst<fst::StdArc>* decode_fst = fst::ReadFstKaldiGeneric(fst_rxfilename_);
  cuda_pipeline_.reset(
      new kaldi::cuda_decoder::BatchedThreadedNnet3CudaOnlinePipeline(
          batched_decoder_config_, *decode_fst, am_nnet_, trans_model_));
  delete decode_fst;

  // Loading word syms for text output
  if (word_syms_rxfilename_ != "") {
    if (!(word_syms_ = fst::SymbolTable::ReadText(word_syms_rxfilename_))) {
      std::cerr << "Could not read symbol table from file "
                << word_syms_rxfilename_;
      return kInvalidModelConfig;
    }
  }
  chunk_num_samps_ = cuda_pipeline_->GetNSampsPerChunk();
  chunk_num_bytes_ = chunk_num_samps_ * sizeof(BaseFloat);
  return kSuccess;
}

int Context::Init() {
  return InputOutputSanityCheck() || ReadModelParameters() ||
         InitializeKaldiPipeline();
}

bool Context::CheckPayloadError(const CustomPayload& payload) {
  int err = payload.error_code;
  if (err) std::cerr << "Error: " << CustomErrorString(err) << std::endl;
  return (err != 0);
}

int Context::Execute(const uint32_t payload_cnt, CustomPayload* payloads,
                     CustomGetNextInputFn_t input_fn,
                     CustomGetOutputFn_t output_fn) {
  // kaldi::Timer timer;
  if (payload_cnt > num_channels_) return kBatchTooBig;
  // Each payload is a chunk for one sequence
  // Currently using dynamic batcher, not sequence batcher
  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    if (batch_corr_ids_.size() == max_batch_size_) FlushBatch();

    CustomPayload& payload = payloads[pidx];
    if (payload.batch_size != 1) payload.error_code = kTimesteps;
    if (CheckPayloadError(payload)) continue;

    // Get input tensors
    int32_t start, dim, end, ready;
    CorrelationID corr_id;
    const BaseFloat* wave_buffer;
    payload.error_code = GetSequenceInput(
        input_fn, payload.input_context, &corr_id, &start, &ready, &dim, &end,
        &wave_buffer, &wave_byte_buffers_[pidx]);
    if (CheckPayloadError(payload)) continue;
    if (!ready) continue;
    if (dim > chunk_num_samps_) payload.error_code = kChunkTooBig;
    if (CheckPayloadError(payload)) continue;

    kaldi::SubVector<BaseFloat> wave_part(wave_buffer, dim);
    // Initialize corr_id if first chunk
    if (start) cuda_pipeline_->InitCorrID(corr_id);
    // Add to batch
    batch_corr_ids_.push_back(corr_id);
    batch_wave_samples_.push_back(wave_part);
    batch_is_last_chunk_.push_back(end);

    if (end) {
      // If last chunk, set the callback for that seq
      cuda_pipeline_->SetLatticeCallback(
          corr_id, [this, &output_fn, &payloads, pidx,
                    corr_id](kaldi::CompactLattice& clat) {
            std::string output;
            LatticeToString(*word_syms_, clat, &output);
            SetOutputTensor(output, output_fn, payloads[pidx]);
          });
    }
  }
  FlushBatch();
  cuda_pipeline_->WaitForLatticeCallbacks();
  return kSuccess;
}

int Context::FlushBatch() {
  if (!batch_corr_ids_.empty()) {
    cuda_pipeline_->DecodeBatch(batch_corr_ids_, batch_wave_samples_,
                                batch_is_last_chunk_);
    batch_corr_ids_.clear();
    batch_wave_samples_.clear();
    batch_is_last_chunk_.clear();
  }
}

int Context::InputOutputSanityCheck() {
  if (!model_config_.has_sequence_batching()) {
    return kSequenceBatcher;
  }

  auto& batcher = model_config_.sequence_batching();
  if (batcher.control_input_size() != 4) {
    return kModelControl;
  }

  std::set<std::string> control_input_names;
  for (int i = 0; i < 4; ++i)
    control_input_names.insert(batcher.control_input(i).name());
  if (!(control_input_names.erase("START") &&
        control_input_names.erase("END") &&
        control_input_names.erase("CORRID") &&
        control_input_names.erase("READY"))) {
    return kModelControl;
  }

  if (model_config_.input_size() != 2) {
    return kInputOutput;
  }
  if ((model_config_.input(0).dims().size() != 1) ||
      (model_config_.input(0).dims(0) <= 0) ||
      (model_config_.input(1).dims().size() != 1) ||
      (model_config_.input(1).dims(0) != 1)) {
    return kInputOutput;
  }
  chunk_num_samps_ = model_config_.input(0).dims(0);
  chunk_num_bytes_ = chunk_num_samps_ * sizeof(float);

  if ((model_config_.input(0).data_type() != DataType::TYPE_FP32) ||
      (model_config_.input(1).data_type() != DataType::TYPE_INT32)) {
    return kInputOutputDataType;
  }
  if ((model_config_.input(0).name() != "WAV_DATA") ||
      (model_config_.input(1).name() != "WAV_DATA_DIM")) {
    return kInputName;
  }

  if (model_config_.output_size() != 1) {
    return kInputOutput;
  }
  if ((model_config_.output(0).dims().size() != 1) ||
      (model_config_.output(0).dims(0) != 1)) {
    return kInputOutput;
  }
  if (model_config_.output(0).data_type() != DataType::TYPE_STRING) {
    return kInputOutputDataType;
  }
  if (model_config_.output(0).name() != "TEXT") {
    return kOutputName;
  }

  return kSuccess;
}

int Context::GetSequenceInput(CustomGetNextInputFn_t& input_fn,
                              void* input_context, CorrelationID* corr_id,
                              int32_t* start, int32_t* ready, int32_t* dim,
                              int32_t* end, const BaseFloat** wave_buffer,
                              std::vector<uint8_t>* input_buffer) {
  int err;
  //&input_buffer[0]: char pointer -> alias with any types
  // wave_data[0] will holds the struct

  // Get start of sequence tensor
  const void* out;
  err = GetInputTensor(input_fn, input_context, "WAV_DATA_DIM",
                       int32_byte_size_, &byte_buffer_, &out);
  if (err != kSuccess) return err;
  *dim = *reinterpret_cast<const int32_t*>(out);

  err = GetInputTensor(input_fn, input_context, "END", int32_byte_size_,
                       &byte_buffer_, &out);
  if (err != kSuccess) return err;
  *end = *reinterpret_cast<const int32_t*>(out);

  err = GetInputTensor(input_fn, input_context, "START", int32_byte_size_,
                       &byte_buffer_, &out);
  if (err != kSuccess) return err;
  *start = *reinterpret_cast<const int32_t*>(out);

  err = GetInputTensor(input_fn, input_context, "READY", int32_byte_size_,
                       &byte_buffer_, &out);
  if (err != kSuccess) return err;
  *ready = *reinterpret_cast<const int32_t*>(out);

  err = GetInputTensor(input_fn, input_context, "CORRID", int64_byte_size_,
                       &byte_buffer_, &out);
  if (err != kSuccess) return err;
  *corr_id = *reinterpret_cast<const CorrelationID*>(out);

  // Get pointer to speech tensor
  err = GetInputTensor(input_fn, input_context, "WAV_DATA", chunk_num_bytes_,
                       input_buffer, &out);
  if (err != kSuccess) return err;
  *wave_buffer = reinterpret_cast<const BaseFloat*>(out);

  return kSuccess;
}

int Context::SetOutputTensor(const std::string& output,
                             CustomGetOutputFn_t output_fn,
                             CustomPayload payload) {
  uint32_t byte_size_with_size_int = output.size() + sizeof(int32);

  // std::cout << output << std::endl;

  // copy output from best_path to output buffer
  if ((payload.error_code == 0) && (payload.output_cnt > 0)) {
    const char* output_name = payload.required_output_names[0];
    // output buffer
    void* obuffer;
    if (!output_fn(payload.output_context, output_name, output_shape_.size(),
                   &output_shape_[0], byte_size_with_size_int, &obuffer)) {
      payload.error_code = kOutputBuffer;
      return payload.error_code;
    }

    // If no error but the 'obuffer' is returned as nullptr, then
    // skip writing this output.
    if (obuffer != nullptr) {
      // std::cout << "writing " << output << std::endl;
      int32* buffer_as_int = reinterpret_cast<int32*>(obuffer);
      buffer_as_int[0] = output.size();
      memcpy(&buffer_as_int[1], output.data(), output.size());
    }
  }
}
/////////////

extern "C" {

int CustomInitialize(const CustomInitializeData* data, void** custom_context) {
  // Convert the serialized model config to a ModelConfig object.
  ModelConfig model_config;
  if (!model_config.ParseFromString(std::string(
          data->serialized_model_config, data->serialized_model_config_size))) {
    return kInvalidModelConfig;
  }

  // Create the context and validate that the model configuration is
  // something that we can handle.
  Context* context = new Context(std::string(data->instance_name), model_config,
                                 data->gpu_device_id);
  int err = context->Init();
  if (err != kSuccess) {
    return err;
  }

  *custom_context = static_cast<void*>(context);

  return kSuccess;
}

int CustomFinalize(void* custom_context) {
  if (custom_context != nullptr) {
    Context* context = static_cast<Context*>(custom_context);
    delete context;
  }

  return kSuccess;
}

const char* CustomErrorString(void* custom_context, int errcode) {
  return CustomErrorString(errcode);
}

int CustomExecute(void* custom_context, const uint32_t payload_cnt,
                  CustomPayload* payloads, CustomGetNextInputFn_t input_fn,
                  CustomGetOutputFn_t output_fn) {
  if (custom_context == nullptr) {
    return kUnknown;
  }

  Context* context = static_cast<Context*>(custom_context);
  return context->Execute(payload_cnt, payloads, input_fn, output_fn);
}

}  // extern "C"
}
}
}
}  // namespace nvidia::inferenceserver::custom::kaldi_cbe
