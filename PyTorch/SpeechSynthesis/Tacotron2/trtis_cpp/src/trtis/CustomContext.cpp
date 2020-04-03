/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "denoiserLoader.h"
#include "engineCache.h"
#include "tacotron2Loader.h"
#include "utils.h"
#include "waveGlowLoader.h"

#include "CustomContext.hpp"
#include "CharacterMappingReader.hpp"

#include "logging.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "src/core/model_config.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "NvInfer.h"

#include <iostream>

using namespace nvinfer1;
using namespace tts;
using ModelConfig = nvidia::inferenceserver::ModelConfig;
using ModelParameter = nvidia::inferenceserver::ModelParameter;

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char* const ENGINE_EXT = ".eng";
}

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

std::vector<std::string> generateErrorMessages()
{
  std::vector<std::string> msgs(CustomContext::NUM_ERR_CODES);

  msgs[CustomContext::SUCCESS] = "success";
  msgs[CustomContext::BAD_INPUT] = "bad_input";
  msgs[CustomContext::BAD_TENSOR_SIZE] = "bad_tensor_size";

  return msgs;
}

}

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

int CustomContext::create(
    const CustomInitializeData* const data, CustomContext** const customContext)
{
  try {
    CustomContext* context = new CustomContext(data);

    *customContext = context;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create CustomContext: " << e.what() << std::endl;
    return ErrorCode::ERROR;
  }

  return ErrorCode::SUCCESS;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

CustomContext::CustomContext(const CustomInitializeData* const data) :
    TimedObject("CustomContext::execute()"),
    m_name(),
    m_logger(new Logger),
    m_synthesizer(nullptr),
    m_errMessages(generateErrorMessages()),
    m_inputLength(),
    m_outputLength(),
    m_inputHost(),
    m_outputHost(),
    m_reader(CharacterMapping::defaultMapping()),
    m_writer()
{
  ModelConfig modelConfig;
  if (!modelConfig.ParseFromString(std::string(data->serialized_model_config, data->serialized_model_config_size))) {
    throw std::runtime_error("Failed to parse model config.");
  }

  m_name = data->instance_name;

  const std::string enginePath
      = modelConfig.parameters().at("engine_path").string_value();

  try {
    const std::string characterMappingPath
        = modelConfig.parameters().at("mapping_path").string_value();

    m_reader.setCharacterMapping(
        CharacterMappingReader::loadFromFile(characterMappingPath));
    std::cout << "Loaded mapping from '" << characterMappingPath << "'."
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load character mapping due to: " << e.what()
              << std::endl;
    std::cerr << "Using default mapping." << std::endl;
  }

  TRTPtr<IBuilder> builder;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    builder.reset(createInferBuilder(*m_logger));
  }

  EngineCache cache(m_logger);
  std::shared_ptr<Tacotron2Instance> tacotron2 = Tacotron2Loader::load(
      cache,
      *builder,
      enginePath + "/" + Tacotron2Instance::ENGINE_NAME + ENGINE_EXT,
      400,
      false,
      modelConfig.max_batch_size());
  std::shared_ptr<WaveGlowInstance> waveglow = WaveGlowLoader::load(
      cache,
      *builder,
      m_logger,
      enginePath + "/" + WaveGlowInstance::ENGINE_NAME + ENGINE_EXT,
      true,
      modelConfig.max_batch_size());
  std::shared_ptr<DenoiserInstance> denoiser(nullptr);
  if (Utils::parseBool(
          modelConfig.parameters().at("use_denoiser").string_value())) {
    try {
      denoiser = DenoiserLoader::load(
          cache,
          *builder,
          enginePath + "/denoiser.eng",
          true,
          modelConfig.max_batch_size());
    } catch (const std::exception& e) {
      std::cerr << "WARNING: Failed to load denoiser: " << e.what()
                << std::endl;
    }
  }

  m_synthesizer.reset(new SpeechSynthesizer(tacotron2, waveglow, denoiser));

  m_inputLength.resize(m_synthesizer->getMaxBatchSize());
  m_outputLength.resize(m_synthesizer->getMaxBatchSize());
  m_inputHost.resize(
      m_synthesizer->getMaxBatchSize() * m_synthesizer->getMaxInputSize());
  m_outputHost.resize(
      m_synthesizer->getMaxBatchSize() * m_synthesizer->getMaxOutputSize());

  // mark children for timing output
  addChild(m_synthesizer.get());
  addChild(&m_reader);
  addChild(&m_writer);
}


/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/


int CustomContext::execute(
    const int numPayloads,
    CustomPayload* const payloads,
    CustomGetNextInputFn_t inputFn,
    CustomGetOutputFn_t outputFn)
{
  int rv = ErrorCode::SUCCESS;
  try {
    resetTiming();
    startTiming();

    int64_t numSamples = 0;

    for (int payloadIndex = 0; payloadIndex < numPayloads; ++payloadIndex) {
      const custom_payload_struct * payload = payloads+payloadIndex;
      if (payload->input_cnt != 1) {
        throw std::runtime_error(
            "Encountered input count of " + std::to_string(payload->input_cnt)
            + " for payload " + std::to_string(payloadIndex));
      }

      // want input that is just 1 dimension sequence
      if (payload->input_shape_dim_cnts[0] != 1) {
        throw std::runtime_error(
            "Encountered input with "
            + std::to_string(payload->input_shape_dim_cnts[0])
            + " dimensions (only accepts 1).");
      }

      const int batchSize = payloads[payloadIndex].batch_size;
      // copy input to device
      int32_t inputSpacing;
      m_reader.read(
          payload->input_context,
          inputFn,
          m_inputHost.size(),
          batchSize,
          m_inputHost.data(),
          m_inputLength.data(),
          &inputSpacing);

      m_synthesizer->inferFromHost(
          batchSize,
          m_inputHost.data(),
          inputSpacing,
          m_inputLength.data(),
          m_outputHost.data(),
          m_outputLength.data());

      // compute total audio time
      for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        numSamples += m_outputLength[batchIndex];
      }
      m_writer.write(
          payload->output_context,
          outputFn,
          batchSize,
          m_synthesizer->getMaxOutputSize(),
          m_outputHost.data(),
          m_outputLength.data());
    }

    stopTiming();

    const float totalAudioLength = static_cast<float>(numSamples) / 22050.0f;

    std::cout << "Generated " << totalAudioLength << " seconds of 22Khz audio."
              << std::endl;

    std::cout << "GPU Inference Time:" << std::endl;
    printTiming(std::cout, 1);
  } catch (const std::exception& e) {
    std::cerr << "Exception in CustomContext::execute(): " << e.what()
              << std::endl;
    resetTiming();
    rv = ErrorCode::ERROR;
  }

  return rv;
}


const char * CustomContext::errorToString(
    const int error) const
{
  return m_errMessages[error].c_str();
}

std::mutex CustomContext::m_mutex;
