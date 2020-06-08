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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "TRTISClient.hpp"

#include "request_http.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define checkOperation(X) checkOperation_((X), #X " failed.")

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

void checkOperation_(nic::Error err, const std::string& msg)
{
  if (!err.IsOk()) {
    std::ostringstream oss;
    oss << msg << " : " << err;
    throw std::runtime_error(oss.str());
  }
}

std::unique_ptr<nic::InferContext>
createInferContext(const std::string& url, const bool verbose)
{
  const std::string modelName = "tacotron2waveglow";

  std::map<std::string, std::string> http_headers;

  // Create a health context and get the ready and live state of the
  // server.
  std::unique_ptr<nic::ServerHealthContext> health_ctx;

  checkOperation(nic::ServerHealthHttpContext::Create(
      &health_ctx, url, http_headers, verbose));

  bool live, ready;
  checkOperation(health_ctx->GetLive(&live));

  checkOperation(health_ctx->GetReady(&ready));

  if (verbose) {
    std::cout << "Health for model " << modelName << ":" << std::endl;
    std::cout << "Live: " << live << std::endl;
    std::cout << "Ready: " << ready << std::endl;
  }

  // Create a status context and get the status of the model.
  std::unique_ptr<nic::ServerStatusContext> status_ctx;
  checkOperation(nic::ServerStatusHttpContext::Create(
      &status_ctx, url, http_headers, modelName, verbose));

  ni::ServerStatus server_status;
  checkOperation(status_ctx->GetServerStatus(&server_status));

  std::unique_ptr<nic::InferContext> inferContext;
  checkOperation(nic::InferHttpContext::Create(
      &inferContext,
      url,
      http_headers,
      modelName,
      -1 /* model_version */,
      verbose));

  return inferContext;
}

void setBatchSize(nic::InferContext& inferContext, const int batchSize)
{
  // Set the context options to do batch-size 1 requests. Also request
  // that all output tensors be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  checkOperation(nic::InferContext::Options::Create(&options));

  options->SetBatchSize(batchSize);
  for (const auto& output : inferContext.Outputs()) {
    options->AddRawResult(output);
  }

  checkOperation(inferContext.SetRunOptions(*options));
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

TRTISClient::TRTISClient(const std::string& url) : m_url(url)
{
  // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

std::vector<std::vector<float>> TRTISClient::execute(
    const std::vector<std::string>& input,
    int targetBatchSize,
    const bool verbose)
{
  size_t inputIdx = 0;

  nic::Error err;

  // Create the inference context for the model.
  std::unique_ptr<nic::InferContext> inferContext
      = createInferContext(m_url, verbose);
  const int maxBatchSize = inferContext->MaxBatchSize();

  if (targetBatchSize == 0) {
    targetBatchSize = maxBatchSize;
  } else if (maxBatchSize < targetBatchSize) {
    throw std::runtime_error(
        "Request batch size is greater than context can "
        "handle:"
        + std::to_string(targetBatchSize) + " / "
        + std::to_string(maxBatchSize));
  }

  // allocate vectors
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;

  std::vector<std::vector<float>> output;

  // loop until we've handle all input
  while (inputIdx < input.size()) {
    const int batchSize
        = std::min(static_cast<int>(input.size() - inputIdx), targetBatchSize);

    setBatchSize(*inferContext, batchSize);

    // create input tensors
    std::shared_ptr<nic::InferContext::Input> inputDataTensor;
    checkOperation(inferContext->GetInput("INPUT", &inputDataTensor));
    checkOperation(inputDataTensor->Reset());
    checkOperation(inputDataTensor->SetShape(std::vector<int64_t>{batchSize}));
    // queue up batch items
    checkOperation(inputDataTensor->SetFromString(std::vector<std::string>(
        input.begin() + inputIdx, input.begin() + inputIdx + batchSize)));

    // execute synchronously
    checkOperation(inferContext->Run(&results));
    if (results.size() != 2) {
      throw std::runtime_error(
          "Got invalid number of tensor back: " + std::to_string(results.size())
          + ", but expected 2.");
    }

    const uint8_t* resultBytes;
    size_t resultSize;
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
      checkOperation(
          results["OUTPUT"]->GetRaw(batchIndex, &resultBytes, &resultSize));
      const float* const wavData = reinterpret_cast<const float*>(resultBytes);
      const size_t wavSize = resultSize / sizeof(*wavData);

      checkOperation(results["OUTPUT_LENGTH"]->GetRaw(
          batchIndex, &resultBytes, &resultSize));
      const int32_t* const wavLength
          = reinterpret_cast<const int32_t*>(resultBytes);
      const size_t numLengths = resultSize / sizeof(*wavLength);

      if (numLengths != 1) {
        throw std::runtime_error(
            "Got back output with multiple lengths: "
            + std::to_string(numLengths));
      } else if (*wavLength > wavSize) {
        throw std::runtime_error(
            "Got sample length greater than tensor size: "
            + std::to_string(*wavLength) + "/" + std::to_string(wavSize));
      }

      output.emplace_back(std::vector<float>(wavData, wavData + *wavLength));
    }

    inputIdx += batchSize;
  }

  return output;
}

int TRTISClient::getMaxBatchSize() const
{
  std::unique_ptr<nic::InferContext> inferContext
      = createInferContext(m_url, false);
  return inferContext->MaxBatchSize();
}
