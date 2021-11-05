// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#define HAVE_CUDA 1  // Loading Kaldi headers with GPU

#include <triton/backend/backend_common.h>

#include <cfloat>
#include <chrono>
#include <sstream>
#include <thread>

#include "cudadecoder/batched-threaded-nnet3-cuda-online-pipeline.h"
#include "fstext/fstext-lib.h"
#include "kaldi-backend-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-utils.h"
#include "util/kaldi-thread.h"

using kaldi::BaseFloat;

namespace ni = triton::common;
namespace nib = triton::backend;

namespace {

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                              \
  do {                                                                       \
    TRITONSERVER_Error* rarie_err__ = (X);                                   \
    if (rarie_err__ != nullptr) {                                            \
      TRITONBACKEND_Response* rarie_response__ = nullptr;                    \
      LOG_IF_ERROR(TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),    \
                   "failed to create response");                             \
      if (rarie_response__ != nullptr) {                                     \
        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(                             \
                         rarie_response__,                                   \
                         TRITONSERVER_RESPONSE_COMPLETE_FINAL, rarie_err__), \
                     "failed to send error response");                       \
      }                                                                      \
      TRITONSERVER_ErrorDelete(rarie_err__);                                 \
      return;                                                                \
    }                                                                        \
  } while (false)

#define RESPOND_FACTORY_AND_RETURN_IF_ERROR(FACTORY, X)                       \
  do {                                                                        \
    TRITONSERVER_Error* rfarie_err__ = (X);                                   \
    if (rfarie_err__ != nullptr) {                                            \
      TRITONBACKEND_Response* rfarie_response__ = nullptr;                    \
      LOG_IF_ERROR(                                                           \
          TRITONBACKEND_ResponseNewFromFactory(&rfarie_response__, FACTORY),  \
          "failed to create response");                                       \
      if (rfarie_response__ != nullptr) {                                     \
        LOG_IF_ERROR(TRITONBACKEND_ResponseSend(                              \
                         rfarie_response__,                                   \
                         TRITONSERVER_RESPONSE_COMPLETE_FINAL, rfarie_err__), \
                     "failed to send error response");                        \
      }                                                                       \
      TRITONSERVER_ErrorDelete(rfarie_err__);                                 \
      return;                                                                 \
    }                                                                         \
  } while (false)

//
// ResponseOutput
//
// Bit flags for desired response outputs
//
enum ResponseOutput {
  kResponseOutputRawLattice = 1 << 0,
  kResponseOutputText = 1 << 1,
  kResponseOutputCTM = 1 << 2
};

//
// ModelParams
//
// The parameters parsed from the model configuration.
//
struct ModelParams {
  // Model paths
  std::string nnet3_rxfilename;
  std::string fst_rxfilename;
  std::string word_syms_rxfilename;
  std::string lattice_postprocessor_rxfilename;

  // Filenames
  std::string config_filename;

  uint64_t max_batch_size;
  int num_channels;
  int num_worker_threads;

  int use_tensor_cores;
  float beam;
  float lattice_beam;
  int max_active;
  int frame_subsampling_factor;
  float acoustic_scale;
  int main_q_capacity;
  int aux_q_capacity;

  int chunk_num_bytes;
  int chunk_num_samps;
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Validate and parse the model configuration
  TRITONSERVER_Error* ValidateModelConfig();

  // Obtain the parameters parsed from the model configuration
  const ModelParams* Parameters() { return &model_params_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model,
             ni::TritonJson::Value&& model_config);

  TRITONBACKEND_Model* triton_model_;
  ni::TritonJson::Value model_config_;

  ModelParams model_params_;
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  ni::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  *state = new ModelState(triton_model, std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model,
                       ni::TritonJson::Value&& model_config)
    : triton_model_(triton_model), model_config_(std::move(model_config)) {}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  // We have the json DOM for the model configuration...
  ni::TritonJson::WriteBuffer buffer;
  RETURN_AND_LOG_IF_ERROR(model_config_.PrettyWrite(&buffer),
                          "failed to pretty write model configuration");
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  RETURN_AND_LOG_IF_ERROR(model_config_.MemberAsUInt(
                              "max_batch_size", &model_params_.max_batch_size),
                          "failed to get max batch size");

  ni::TritonJson::Value batcher;
  RETURN_ERROR_IF_FALSE(
      model_config_.Find("sequence_batching", &batcher),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must configure sequence batcher"));
  ni::TritonJson::Value control_inputs;
  RETURN_AND_LOG_IF_ERROR(
      batcher.MemberAsArray("control_input", &control_inputs),
      "failed to read control input array");
  std::set<std::string> control_input_names;
  for (uint32_t i = 0; i < control_inputs.ArraySize(); i++) {
    ni::TritonJson::Value control_input;
    RETURN_AND_LOG_IF_ERROR(control_inputs.IndexAsObject(i, &control_input),
                            "failed to get control input");
    std::string control_input_name;
    RETURN_AND_LOG_IF_ERROR(
        control_input.MemberAsString("name", &control_input_name),
        "failed to get control input name");
    control_input_names.insert(control_input_name);
  }

  RETURN_ERROR_IF_FALSE(
      (control_input_names.erase("START") && control_input_names.erase("END") &&
       control_input_names.erase("CORRID") &&
       control_input_names.erase("READY")),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("missing control input names in the model configuration"));

  // Check the Model Transaction Policy
  ni::TritonJson::Value txn_policy;
  RETURN_ERROR_IF_FALSE(
      model_config_.Find("model_transaction_policy", &txn_policy),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must specify a transaction policy"));
  bool is_decoupled;
  RETURN_AND_LOG_IF_ERROR(txn_policy.MemberAsBool("decoupled", &is_decoupled),
                          "failed to read the decouled txn policy");
  RETURN_ERROR_IF_FALSE(
      is_decoupled, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must use decoupled transaction policy"));

  // Check the Inputs and Outputs
  ni::TritonJson::Value inputs, outputs;
  RETURN_AND_LOG_IF_ERROR(model_config_.MemberAsArray("input", &inputs),
                          "failed to read input array");
  RETURN_AND_LOG_IF_ERROR(model_config_.MemberAsArray("output", &outputs),
                          "failed to read output array");

  // There must be 2 inputs and 3 outputs.
  RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected 2 inputs, got ") +
                            std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(outputs.ArraySize() == 3,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected 3 outputs, got ") +
                            std::to_string(outputs.ArraySize()));

  // Here we rely on the model configuation listing the inputs and
  // outputs in a specific order, which we shouldn't really require...
  // TODO use sets and loops
  ni::TritonJson::Value in0, in1, out0, out1, out2;
  RETURN_AND_LOG_IF_ERROR(inputs.IndexAsObject(0, &in0),
                          "failed to get the first input");
  RETURN_AND_LOG_IF_ERROR(inputs.IndexAsObject(1, &in1),
                          "failed to get the second input");
  RETURN_AND_LOG_IF_ERROR(outputs.IndexAsObject(0, &out0),
                          "failed to get the first output");
  RETURN_AND_LOG_IF_ERROR(outputs.IndexAsObject(1, &out1),
                          "failed to get the second output");
  RETURN_AND_LOG_IF_ERROR(outputs.IndexAsObject(2, &out2),
                          "failed to get the third output");

  // Check tensor names
  std::string in0_name, in1_name, out0_name, out1_name, out2_name;
  RETURN_AND_LOG_IF_ERROR(in0.MemberAsString("name", &in0_name),
                          "failed to get the first input name");
  RETURN_AND_LOG_IF_ERROR(in1.MemberAsString("name", &in1_name),
                          "failed to get the second input name");
  RETURN_AND_LOG_IF_ERROR(out0.MemberAsString("name", &out0_name),
                          "failed to get the first output name");
  RETURN_AND_LOG_IF_ERROR(out1.MemberAsString("name", &out1_name),
                          "failed to get the second output name");
  RETURN_AND_LOG_IF_ERROR(out2.MemberAsString("name", &out2_name),
                          "failed to get the third output name");

  RETURN_ERROR_IF_FALSE(
      in0_name == "WAV_DATA", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first input tensor name to be WAV_DATA, got ") +
          in0_name);
  RETURN_ERROR_IF_FALSE(
      in1_name == "WAV_DATA_DIM", TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "expected second input tensor name to be WAV_DATA_DIM, got ") +
          in1_name);
  RETURN_ERROR_IF_FALSE(
      out0_name == "RAW_LATTICE", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected first output tensor name to be RAW_LATTICE, got ") +
          out0_name);
  RETURN_ERROR_IF_FALSE(
      out1_name == "TEXT", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second output tensor name to be TEXT, got ") +
          out1_name);
  RETURN_ERROR_IF_FALSE(
      out2_name == "CTM", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected second output tensor name to be CTM, got ") +
          out2_name);

  // Check shapes
  std::vector<int64_t> in0_shape, in1_shape, out0_shape, out1_shape;
  RETURN_AND_LOG_IF_ERROR(nib::ParseShape(in0, "dims", &in0_shape),
                          " first input shape");
  RETURN_AND_LOG_IF_ERROR(nib::ParseShape(in1, "dims", &in1_shape),
                          " second input shape");
  RETURN_AND_LOG_IF_ERROR(nib::ParseShape(out0, "dims", &out0_shape),
                          " first output shape");
  RETURN_AND_LOG_IF_ERROR(nib::ParseShape(out1, "dims", &out1_shape),
                          " second ouput shape");

  RETURN_ERROR_IF_FALSE(
      in0_shape.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAV_DATA shape to have one dimension, got ") +
          nib::ShapeToString(in0_shape));
  RETURN_ERROR_IF_FALSE(
      in0_shape[0] > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAV_DATA shape to be greater than 0, got ") +
          nib::ShapeToString(in0_shape));
  model_params_.chunk_num_samps = in0_shape[0];
  model_params_.chunk_num_bytes = model_params_.chunk_num_samps * sizeof(float);

  RETURN_ERROR_IF_FALSE(
      ((in1_shape.size() == 1) && (in1_shape[0] == 1)),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAV_DATA_DIM shape to be [1], got ") +
          nib::ShapeToString(in1_shape));
  RETURN_ERROR_IF_FALSE(
      ((out0_shape.size() == 1) && (out0_shape[0] == 1)),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected RAW_LATTICE shape to be [1], got ") +
          nib::ShapeToString(out0_shape));
  RETURN_ERROR_IF_FALSE(((out1_shape.size() == 1) && (out1_shape[0] == 1)),
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected TEXT shape to be [1], got ") +
                            nib::ShapeToString(out1_shape));

  // Check datatypes
  std::string in0_dtype, in1_dtype, out0_dtype, out1_dtype;
  RETURN_AND_LOG_IF_ERROR(in0.MemberAsString("data_type", &in0_dtype),
                          "first input data type");
  RETURN_AND_LOG_IF_ERROR(in1.MemberAsString("data_type", &in1_dtype),
                          "second input datatype");
  RETURN_AND_LOG_IF_ERROR(out0.MemberAsString("data_type", &out0_dtype),
                          "first output datatype");
  RETURN_AND_LOG_IF_ERROR(out1.MemberAsString("data_type", &out1_dtype),
                          "second output datatype");

  RETURN_ERROR_IF_FALSE(
      in0_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected IN datatype to be INT32, got ") + in0_dtype);
  RETURN_ERROR_IF_FALSE(
      in1_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected DELAY datatype to be UINT32, got ") + in1_dtype);
  RETURN_ERROR_IF_FALSE(
      out0_dtype == "TYPE_STRING", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected WAIT datatype to be UINT32, got ") + out0_dtype);
  RETURN_ERROR_IF_FALSE(
      out1_dtype == "TYPE_STRING", TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected OUT datatype to be INT32, got ") + out1_dtype);

  // Validate and set parameters
  ni::TritonJson::Value params;
  RETURN_ERROR_IF_FALSE(
      (model_config_.Find("parameters", &params)),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("missing parameters in the model configuration"));
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "config_filename",
                                             &(model_params_.config_filename)),
                          "config_filename");
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "use_tensor_cores",
                                             &(model_params_.use_tensor_cores)),
                          "cuda use tensor cores");
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "main_q_capacity",
                                             &(model_params_.main_q_capacity)),
                          "cuda use tensor cores");
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "aux_q_capacity",
                                             &(model_params_.aux_q_capacity)),
                          "cuda use tensor cores");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "beam", &(model_params_.beam)), "beam");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "lattice_beam", &(model_params_.lattice_beam)),
      "lattice beam");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "max_active", &(model_params_.max_active)),
      "max active");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "frame_subsampling_factor",
                         &(model_params_.frame_subsampling_factor)),
      "frame_subsampling_factor");
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "acoustic_scale",
                                             &(model_params_.acoustic_scale)),
                          "acoustic_scale");
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "nnet3_rxfilename",
                                             &(model_params_.nnet3_rxfilename)),
                          "nnet3_rxfilename");
  RETURN_AND_LOG_IF_ERROR(nib::ReadParameter(params, "fst_rxfilename",
                                             &(model_params_.fst_rxfilename)),
                          "fst_rxfilename");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "word_syms_rxfilename",
                         &(model_params_.word_syms_rxfilename)),
      "word_syms_rxfilename");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "num_worker_threads",
                         &(model_params_.num_worker_threads)),
      "num_worker_threads");
  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "num_channels", &(model_params_.num_channels)),
      "num_channels");

  RETURN_AND_LOG_IF_ERROR(
      nib::ReadParameter(params, "lattice_postprocessor_rxfilename",
                         &(model_params_.lattice_postprocessor_rxfilename)),
      "(optional) lattice postprocessor config file");

  model_params_.max_batch_size = std::max<int>(model_params_.max_batch_size, 1);
  model_params_.num_channels = std::max<int>(model_params_.num_channels, 1);

  // Sanity checks
  RETURN_ERROR_IF_FALSE(
      model_params_.beam > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected parameter \"beam\" to be greater than 0, got ") +
          std::to_string(model_params_.beam));
  RETURN_ERROR_IF_FALSE(
      model_params_.lattice_beam > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "expected parameter \"lattice_beam\" to be greater than 0, got ") +
          std::to_string(model_params_.lattice_beam));
  RETURN_ERROR_IF_FALSE(
      model_params_.max_active > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "expected parameter \"max_active\" to be greater than 0, got ") +
          std::to_string(model_params_.max_active));
  RETURN_ERROR_IF_FALSE(model_params_.main_q_capacity >= -1,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected parameter \"main_q_capacity\" to "
                                    "be greater than or equal to -1, got ") +
                            std::to_string(model_params_.main_q_capacity));
  RETURN_ERROR_IF_FALSE(model_params_.aux_q_capacity >= -1,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected parameter \"aux_q_capacity\" to "
                                    "be greater than or equal to -1, got ") +
                            std::to_string(model_params_.aux_q_capacity));
  RETURN_ERROR_IF_FALSE(
      model_params_.acoustic_scale > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "expected parameter \"acoustic_scale\" to be greater than 0, got ") +
          std::to_string(model_params_.acoustic_scale));
  RETURN_ERROR_IF_FALSE(
      model_params_.num_worker_threads >= -1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected parameter \"num_worker_threads\" to be greater "
                  "than or equal to -1, got ") +
          std::to_string(model_params_.num_worker_threads));
  RETURN_ERROR_IF_FALSE(
      model_params_.num_channels > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "expected parameter \"num_channels\" to be greater than 0, got ") +
          std::to_string(model_params_.num_channels));

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  ~ModelInstanceState();

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance() {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Initialize this object
  TRITONSERVER_Error* Init();

  // Initialize kaldi pipeline with this object
  TRITONSERVER_Error* InitializeKaldiPipeline();

  // Prepares the requests for kaldi pipeline
  void PrepareRequest(TRITONBACKEND_Request* request, uint32_t slot_idx);

  // Executes the batch on the decoder
  void FlushBatch();

  // Waits for all pipeline callbacks to complete
  void WaitForLatticeCallbacks();

 private:
  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance,
                     const char* name,
                     const TRITONSERVER_InstanceGroupKind kind,
                     const int32_t device_id);

  TRITONSERVER_Error* GetSequenceInput(TRITONBACKEND_Request* request,
                                       int32_t* start, int32_t* ready,
                                       int32_t* dim, int32_t* end,
                                       uint64_t* corr_id,
                                       const BaseFloat** wave_buffer,
                                       std::vector<uint8_t>* input_buffer);

  void DeliverPartialResponse(const std::string& text,
                              TRITONBACKEND_ResponseFactory* response_factory,
                              uint8_t response_outputs);
  void DeliverResponse(
      std::vector<kaldi::cuda_decoder::CudaPipelineResult>& results,
      uint64_t corr_id, TRITONBACKEND_ResponseFactory* response_factory,
      uint8_t response_outputs);
  void SetPartialOutput(const std::string& text,
                        TRITONBACKEND_ResponseFactory* response_factory,
                        TRITONBACKEND_Response* response);
  void SetOutput(std::vector<kaldi::cuda_decoder::CudaPipelineResult>& results,
                 uint64_t corr_id, const std::string& output_name,
                 TRITONBACKEND_ResponseFactory* response_factory,
                 TRITONBACKEND_Response* response);

  void SetOutputBuffer(const std::string& out_bytes,
                       TRITONBACKEND_Response* response,
                       TRITONBACKEND_Output* response_output);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  std::mutex partial_resfactory_mu_;
  std::unordered_map<uint64_t,
                     std::queue<std::shared_ptr<TRITONBACKEND_ResponseFactory>>>
      partial_responsefactory_;
  std::vector<uint64_t> batch_corr_ids_;
  std::vector<kaldi::SubVector<kaldi::BaseFloat>> batch_wave_samples_;
  std::vector<bool> batch_is_first_chunk_;
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

  std::vector<uint8_t> byte_buffer_;
  std::vector<std::vector<uint8_t>> wave_byte_buffers_;

  std::vector<int64_t> output_shape_;
  std::vector<std::string> request_outputs_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(model_state, triton_model_instance,
                                  instance_name, instance_kind, instance_id);
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state),
      triton_model_instance_(triton_model_instance),
      name_(name),
      kind_(kind),
      device_id_(device_id) {}

ModelInstanceState::~ModelInstanceState() { delete word_syms_; }

TRITONSERVER_Error* ModelInstanceState::Init() {
  const ModelParams* model_params = model_state_->Parameters();

  chunk_num_samps_ = model_params->chunk_num_samps;
  chunk_num_bytes_ = model_params->chunk_num_bytes;


  {
    std::ostringstream usage_str;
    usage_str << "Parsing config from " << "from '" << model_params->config_filename << "'";
    kaldi::ParseOptions po(usage_str.str().c_str());
    batched_decoder_config_.Register(&po);
    po.DisableOption("cuda-decoder-copy-threads");
    po.DisableOption("cuda-worker-threads");
    po.DisableOption("max-active");
    po.DisableOption("max-batch-size");
    po.DisableOption("num-channels");
    po.ReadConfigFile(model_params->config_filename);
  }
  kaldi::CuDevice::EnableTensorCores(bool(model_params->use_tensor_cores));

  batched_decoder_config_.compute_opts.frame_subsampling_factor =
      model_params->frame_subsampling_factor;
  batched_decoder_config_.compute_opts.acoustic_scale =
      model_params->acoustic_scale;
  batched_decoder_config_.decoder_opts.default_beam = model_params->beam;
  batched_decoder_config_.decoder_opts.lattice_beam =
      model_params->lattice_beam;
  batched_decoder_config_.decoder_opts.max_active = model_params->max_active;
  batched_decoder_config_.num_worker_threads = model_params->num_worker_threads;
  batched_decoder_config_.max_batch_size = model_params->max_batch_size;
  batched_decoder_config_.num_channels = model_params->num_channels;
  batched_decoder_config_.decoder_opts.main_q_capacity =
      model_params->main_q_capacity;
  batched_decoder_config_.decoder_opts.aux_q_capacity =
      model_params->aux_q_capacity;

  auto feature_config = batched_decoder_config_.feature_opts;
  kaldi::OnlineNnet2FeaturePipelineInfo feature_info(feature_config);
  sample_freq_ = feature_info.mfcc_opts.frame_opts.samp_freq;
  BaseFloat frame_shift = feature_info.FrameShiftInSeconds();
  seconds_per_chunk_ = chunk_num_samps_ / sample_freq_;

  int samp_per_frame = static_cast<int>(sample_freq_ * frame_shift);
  float n_input_framesf = chunk_num_samps_ / samp_per_frame;
  RETURN_ERROR_IF_FALSE(
      (n_input_framesf == std::floor(n_input_framesf)),
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("WAVE_DATA dim must be a multiple fo samples per frame (") +
          std::to_string(samp_per_frame) + std::string(")"));
  int n_input_frames = static_cast<int>(std::floor(n_input_framesf));
  batched_decoder_config_.compute_opts.frames_per_chunk = n_input_frames;

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::InitializeKaldiPipeline() {
  const ModelParams* model_params = model_state_->Parameters();

  batch_corr_ids_.reserve(model_params->max_batch_size);
  batch_wave_samples_.reserve(model_params->max_batch_size);
  batch_is_first_chunk_.reserve(model_params->max_batch_size);
  batch_is_last_chunk_.reserve(model_params->max_batch_size);
  wave_byte_buffers_.resize(model_params->max_batch_size);
  for (auto& wbb : wave_byte_buffers_) {
    wbb.resize(chunk_num_bytes_);
  }
  output_shape_ = {1, 1};
  kaldi::g_cuda_allocator.SetOptions(kaldi::g_allocator_options);
  kaldi::CuDevice::Instantiate()
      .SelectAndInitializeGpuIdWithExistingCudaContext(device_id_);
  kaldi::CuDevice::Instantiate().AllowMultithreading();

  // Loading models
  {
    bool binary;
    kaldi::Input ki(model_params->nnet3_rxfilename, &binary);
    trans_model_.Read(ki.Stream(), binary);
    am_nnet_.Read(ki.Stream(), binary);

    kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
    kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
    kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(),
                                &(am_nnet_.GetNnet()));
  }
  fst::Fst<fst::StdArc>* decode_fst =
      fst::ReadFstKaldiGeneric(model_params->fst_rxfilename);
  cuda_pipeline_.reset(
      new kaldi::cuda_decoder::BatchedThreadedNnet3CudaOnlinePipeline(
          batched_decoder_config_, *decode_fst, am_nnet_, trans_model_));
  delete decode_fst;

  // Loading word syms for text output
  if (model_params->word_syms_rxfilename != "") {
    RETURN_ERROR_IF_FALSE(
        (word_syms_ =
             fst::SymbolTable::ReadText(model_params->word_syms_rxfilename)),
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("could not read symbol table from file ") +
            model_params->word_syms_rxfilename);
    cuda_pipeline_->SetSymbolTable(*word_syms_);
  }

  // Load lattice postprocessor, required if using CTM
  if (!model_params->lattice_postprocessor_rxfilename.empty()) {
    LoadAndSetLatticePostprocessor(
        model_params->lattice_postprocessor_rxfilename, cuda_pipeline_.get());
  }
  chunk_num_samps_ = cuda_pipeline_->GetNSampsPerChunk();
  chunk_num_bytes_ = chunk_num_samps_ * sizeof(BaseFloat);

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::GetSequenceInput(
    TRITONBACKEND_Request* request, int32_t* start, int32_t* ready,
    int32_t* dim, int32_t* end, uint64_t* corr_id,
    const BaseFloat** wave_buffer, std::vector<uint8_t>* input_buffer) {
  size_t dim_bsize = sizeof(*dim);
  RETURN_IF_ERROR(nib::ReadInputTensor(
      request, "WAV_DATA_DIM", reinterpret_cast<char*>(dim), &dim_bsize));

  size_t end_bsize = sizeof(*end);
  RETURN_IF_ERROR(nib::ReadInputTensor(
      request, "END", reinterpret_cast<char*>(end), &end_bsize));

  size_t start_bsize = sizeof(*start);
  RETURN_IF_ERROR(nib::ReadInputTensor(
      request, "START", reinterpret_cast<char*>(start), &start_bsize));

  size_t ready_bsize = sizeof(*ready);
  RETURN_IF_ERROR(nib::ReadInputTensor(
      request, "READY", reinterpret_cast<char*>(ready), &ready_bsize));

  size_t corrid_bsize = sizeof(*corr_id);
  RETURN_IF_ERROR(nib::ReadInputTensor(
      request, "CORRID", reinterpret_cast<char*>(corr_id), &corrid_bsize));

  // Get pointer to speech tensor
  size_t wavdata_bsize = input_buffer->size();
  RETURN_IF_ERROR(nib::ReadInputTensor(
      request, "WAV_DATA", reinterpret_cast<char*>(input_buffer->data()),
      &wavdata_bsize));
  *wave_buffer = reinterpret_cast<const BaseFloat*>(input_buffer->data());

  return nullptr;
}

void ModelInstanceState::PrepareRequest(TRITONBACKEND_Request* request,
                                        uint32_t slot_idx) {
  const ModelParams* model_params = model_state_->Parameters();

  if (batch_corr_ids_.size() == (uint32_t)model_params->max_batch_size) {
    FlushBatch();
  }

  int32_t start, dim, end, ready;
  uint64_t corr_id;
  const BaseFloat* wave_buffer;

  if (slot_idx >= (uint32_t)model_params->max_batch_size) {
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                       "slot_idx exceeded"));
  }
  RESPOND_AND_RETURN_IF_ERROR(
      request, GetSequenceInput(request, &start, &ready, &dim, &end, &corr_id,
                                &wave_buffer, &wave_byte_buffers_[slot_idx]));

  uint32_t output_count;
  RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestOutputCount(request, &output_count));

  uint8_t response_outputs = 0;
  int kaldi_result_type = 0;
  for (uint32_t index = 0; index < output_count; index++) {
    const char* output_name;
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONBACKEND_RequestOutputName(request, index, &output_name));
    std::string output_name_str = output_name;
    if (output_name_str == "RAW_LATTICE") {
      response_outputs |= kResponseOutputRawLattice;
      kaldi_result_type |=
          kaldi::cuda_decoder::CudaPipelineResult::RESULT_TYPE_LATTICE;
    } else if (output_name_str == "TEXT") {
      response_outputs |= kResponseOutputText;
      kaldi_result_type |=
          kaldi::cuda_decoder::CudaPipelineResult::RESULT_TYPE_LATTICE;
    } else if (output_name_str == "CTM") {
      response_outputs |= kResponseOutputCTM;
      kaldi_result_type |=
          kaldi::cuda_decoder::CudaPipelineResult::RESULT_TYPE_CTM;
    } else {
      TRITONSERVER_LogMessage(
          TRITONSERVER_LOG_WARN, __FILE__, __LINE__,
          ("unrecognized requested output " + output_name_str).c_str());
    }
  }

  if (dim > chunk_num_samps_) {
    RESPOND_AND_RETURN_IF_ERROR(
        request,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "a chunk cannot contain more samples than the WAV_DATA dimension"));
  }

  if (!ready) {
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                       "request is not yet ready"));
  }

  // Initialize corr_id if first chunk
  if (start) {
    if (!cuda_pipeline_->TryInitCorrID(corr_id)) {
      RESPOND_AND_RETURN_IF_ERROR(
          request, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                         "failed to start cuda pipeline"));
    }

    {
      std::lock_guard<std::mutex> lock_partial_resfactory(
          partial_resfactory_mu_);
      cuda_pipeline_->SetBestPathCallback(
          corr_id, [this, corr_id](const std::string& str, bool partial,
                                   bool endpoint_detected) {
            // Bestpath callbacks are synchronous in regards to each correlation
            // ID, so the lock is only needed for acquiring a reference to the
            // queue.
            std::unique_lock<std::mutex> lock_partial_resfactory(
                partial_resfactory_mu_);
            auto& resfactory_queue = partial_responsefactory_.at(corr_id);
            if (!partial) {
              if (!endpoint_detected) {
                // while (!resfactory_queue.empty()) {
                //   auto response_factory = resfactory_queue.front();
                //   resfactory_queue.pop();
                //   if (response_factory != nullptr) {
                //     LOG_IF_ERROR(
                //         TRITONBACKEND_ResponseFactoryDelete(response_factory),
                //         "error deleting response factory");
                //   }
                // }
                partial_responsefactory_.erase(corr_id);
              }
              return;
            }
            if (resfactory_queue.empty()) {
              TRITONSERVER_LogMessage(
                  TRITONSERVER_LOG_WARN, __FILE__, __LINE__,
                  "response factory queue unexpectedly empty");
              return;
            }

            auto response_factory = resfactory_queue.front();
            resfactory_queue.pop();
            lock_partial_resfactory.unlock();
            if (response_factory == nullptr) return;

            DeliverPartialResponse(str, response_factory.get(),
                                   kResponseOutputText);
          });
      partial_responsefactory_.emplace(
          corr_id,
          std::queue<std::shared_ptr<TRITONBACKEND_ResponseFactory>>());
    }
  }

  kaldi::SubVector<BaseFloat> wave_part(wave_buffer, dim);

  // Add to batch
  batch_corr_ids_.push_back(corr_id);
  batch_wave_samples_.push_back(wave_part);
  batch_is_first_chunk_.push_back(start);
  batch_is_last_chunk_.push_back(end);

  TRITONBACKEND_ResponseFactory* response_factory_ptr;
  RESPOND_AND_RETURN_IF_ERROR(request, TRITONBACKEND_ResponseFactoryNew(
                                           &response_factory_ptr, request));
  std::shared_ptr<TRITONBACKEND_ResponseFactory> response_factory(
      response_factory_ptr, [](TRITONBACKEND_ResponseFactory* f) {
        LOG_IF_ERROR(TRITONBACKEND_ResponseFactoryDelete(f),
                     "failed deleting response factory");
      });

  if (end) {
    auto segmented_lattice_callback_fn =
        [this, response_factory, response_outputs,
         corr_id](kaldi::cuda_decoder::SegmentedLatticeCallbackParams& params) {
          DeliverResponse(params.results, corr_id, response_factory.get(),
                          response_outputs);
        };
    cuda_pipeline_->SetLatticeCallback(corr_id, segmented_lattice_callback_fn,
                                       kaldi_result_type);
  } else if (response_outputs & kResponseOutputText) {
    std::lock_guard<std::mutex> lock_partial_resfactory(partial_resfactory_mu_);
    auto& resfactory_queue = partial_responsefactory_.at(corr_id);
    resfactory_queue.push(response_factory);
  } else {
    {
      std::lock_guard<std::mutex> lock_partial_resfactory(
          partial_resfactory_mu_);
      auto& resfactory_queue = partial_responsefactory_.at(corr_id);
      resfactory_queue.emplace(nullptr);
    }

    // Mark the response complete without sending any responses
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseFactorySendFlags(
            response_factory.get(), TRITONSERVER_RESPONSE_COMPLETE_FINAL),
        "failed sending final response");
  }
}

void ModelInstanceState::FlushBatch() {
  if (!batch_corr_ids_.empty()) {
    cuda_pipeline_->DecodeBatch(batch_corr_ids_, batch_wave_samples_,
                                batch_is_first_chunk_, batch_is_last_chunk_);
    batch_corr_ids_.clear();
    batch_wave_samples_.clear();
    batch_is_first_chunk_.clear();
    batch_is_last_chunk_.clear();
  }
}

void ModelInstanceState::WaitForLatticeCallbacks() {
  cuda_pipeline_->WaitForLatticeCallbacks();
}

void ModelInstanceState::DeliverPartialResponse(
    const std::string& text, TRITONBACKEND_ResponseFactory* response_factory,
    uint8_t response_outputs) {
  if (response_outputs & kResponseOutputText) {
    TRITONBACKEND_Response* response;
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        response_factory,
        TRITONBACKEND_ResponseNewFromFactory(&response, response_factory));
    SetPartialOutput(text, response_factory, response);
    LOG_IF_ERROR(TRITONBACKEND_ResponseSend(
                     response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                 "failed sending response");
  } else {
    LOG_IF_ERROR(TRITONBACKEND_ResponseFactorySendFlags(
                     response_factory, TRITONSERVER_RESPONSE_COMPLETE_FINAL),
                 "failed to send final flag for partial result");
  }
}

void ModelInstanceState::DeliverResponse(
    std::vector<kaldi::cuda_decoder::CudaPipelineResult>& results,
    uint64_t corr_id, TRITONBACKEND_ResponseFactory* response_factory,
    uint8_t response_outputs) {
  TRITONBACKEND_Response* response;
  RESPOND_FACTORY_AND_RETURN_IF_ERROR(
      response_factory,
      TRITONBACKEND_ResponseNewFromFactory(&response, response_factory));
  if (response_outputs & kResponseOutputRawLattice) {
    SetOutput(results, corr_id, "RAW_LATTICE", response_factory, response);
  }
  if (response_outputs & kResponseOutputText) {
    SetOutput(results, corr_id, "TEXT", response_factory, response);
  }
  if (response_outputs & kResponseOutputCTM) {
    SetOutput(results, corr_id, "CTM", response_factory, response);
  }
  // Send the response.
  LOG_IF_ERROR(
      TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                                 nullptr /* success */),
      "failed sending response");
}

void ModelInstanceState::SetPartialOutput(
    const std::string& text, TRITONBACKEND_ResponseFactory* response_factory,
    TRITONBACKEND_Response* response) {
  TRITONBACKEND_Output* response_output;
  RESPOND_FACTORY_AND_RETURN_IF_ERROR(
      response_factory, TRITONBACKEND_ResponseOutput(
                            response, &response_output, "TEXT",
                            TRITONSERVER_TYPE_BYTES, &output_shape_[0], 2));
  SetOutputBuffer(text, response, response_output);
}

void ModelInstanceState::SetOutput(
    std::vector<kaldi::cuda_decoder::CudaPipelineResult>& results,
    uint64_t corr_id, const std::string& output_name,
    TRITONBACKEND_ResponseFactory* response_factory,
    TRITONBACKEND_Response* response) {
  TRITONBACKEND_Output* response_output;
  RESPOND_FACTORY_AND_RETURN_IF_ERROR(
      response_factory,
      TRITONBACKEND_ResponseOutput(response, &response_output,
                                   output_name.c_str(), TRITONSERVER_TYPE_BYTES,
                                   &output_shape_[0], 2 /* dims_count */));

  if (output_name.compare("RAW_LATTICE") == 0) {
    assert(!results.empty());
    kaldi::CompactLattice& clat = results[0].GetLatticeResult();

    std::ostringstream oss;
    kaldi::WriteCompactLattice(oss, true, clat);
    SetOutputBuffer(oss.str(), response, response_output);
  } else if (output_name.compare("TEXT") == 0) {
    assert(!results.empty());
    kaldi::CompactLattice& clat = results[0].GetLatticeResult();
    std::string output;
    nib::LatticeToString(*word_syms_, clat, &output);
    SetOutputBuffer(output, response, response_output);
  } else if (output_name.compare("CTM") == 0) {
    std::ostringstream oss;
    MergeSegmentsToCTMOutput(results, std::to_string(corr_id), oss, word_syms_,
                             /* use segment offset*/ false);
    SetOutputBuffer(oss.str(), response, response_output);
  }
}

void ModelInstanceState::SetOutputBuffer(
    const std::string& out_bytes, TRITONBACKEND_Response* response,
    TRITONBACKEND_Output* response_output) {
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;
  uint32_t byte_size_with_size_int = out_bytes.size() + sizeof(int32);
  void* obuffer;  // output buffer
  auto err = TRITONBACKEND_OutputBuffer(
      response_output, &obuffer, byte_size_with_size_int, &actual_memory_type,
      &actual_memory_type_id);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
  }

  int32* buffer_as_int = reinterpret_cast<int32*>(obuffer);
  buffer_as_int[0] = out_bytes.size();
  memcpy(&buffer_as_int[1], out_bytes.data(), out_bytes.size());
}

}  // namespace

/////////////

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());

  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
               " (device " + std::to_string(device_id) + ")")
                  .c_str());

  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  RETURN_IF_ERROR(instance_state->Init());
  RETURN_IF_ERROR(instance_state->InitializeKaldiPipeline());

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: waiting for lattice callbacks");
  instance_state->WaitForLatticeCallbacks();

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelInstanceFinalize: delete instance state");
  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("model instance ") + instance_state->Name() +
               ", executing " + std::to_string(request_count) + " requests")
                  .c_str());

  RETURN_ERROR_IF_FALSE(
      request_count <=
          instance_state->StateForModel()->Parameters()->max_batch_size,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("request count exceeded the provided maximum batch size"));

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Each request is a chunk for one sequence
  // Using the oldest strategy in the sequence batcher ensures that
  // there will only be a single chunk for each sequence.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    instance_state->PrepareRequest(request, r);
  }

  instance_state->FlushBatch();

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request, true /* success */,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                   instance_state->TritonModelInstance(), request_count,
                   exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
               "failed reporting batch request statistics");

  return nullptr;  // success
}

}  // extern "C"
