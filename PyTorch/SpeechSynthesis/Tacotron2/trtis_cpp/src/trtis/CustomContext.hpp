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

#ifndef TT2I_CUSTOMCONTEXT_HPP
#define TT2I_CUSTOMCONTEXT_HPP

#include "CustomInputReader.hpp"
#include "CustomOutputWriter.hpp"
#include "speechSynthesizer.h"
#include "timedObject.h"
#include "logging.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "src/backends/custom/custom.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <memory>
#include <mutex>
#include <string>
#include <vector>

class CustomContext : public tts::TimedObject
{
  public:
    enum ErrorCode
    {
      SUCCESS = 0, // must be 0 as defined by custom.h
      BAD_INPUT,
      BAD_TENSOR_SIZE,
      ERROR,
      NUM_ERR_CODES
    };

    static int create(
        const CustomInitializeData* const data, CustomContext** customContext);

    CustomContext(const CustomInitializeData* const data);

    int execute(
        const int numPayloads,
        CustomPayload* const payloads,
        CustomGetNextInputFn_t inputFn,
        CustomGetOutputFn_t outputFn);

    const char* errorToString(const int error) const;

  private:
  std::string m_name;
  std::shared_ptr<Logger> m_logger;
  std::unique_ptr<tts::SpeechSynthesizer> m_synthesizer;
  std::vector<std::string> m_errMessages;
  std::vector<int32_t> m_inputLength;
  std::vector<int32_t> m_outputLength;
  std::vector<int32_t> m_inputHost;
  std::vector<float> m_outputHost;
  CustomInputReader m_reader;
  CustomOutputWriter m_writer;

  static std::mutex m_mutex;
};

#endif
