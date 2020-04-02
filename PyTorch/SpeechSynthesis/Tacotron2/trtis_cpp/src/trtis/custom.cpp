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

#include "CustomContext.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "src/backends/custom/custom.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <iostream>

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

CustomContext* parseContext(
    void * customContext)
{
  return reinterpret_cast<CustomContext*>(customContext);
}


}

/******************************************************************************
 * BACKEND FUNCTIONS **********************************************************
 *****************************************************************************/

int CustomInitialize(
    const CustomInitializeData* data, void** customContext)
{
  return CustomContext::create(
      data, reinterpret_cast<CustomContext**>(customContext));
}

int CustomFinalize(void* const customContext)
{
  CustomContext * const context = parseContext(customContext);
  delete context;

  return CustomContext::ErrorCode::SUCCESS;
}

const char* CustomErrorString(
    void* const customContext, const int errcode)
{
  CustomContext * const context = parseContext(customContext);

  return context->errorToString(errcode);
}

int CustomExecute(
    void* const customContext,
    const uint32_t numPayloads,
    CustomPayload* const payloads,
    CustomGetNextInputFn_t input_fn,
    CustomGetOutputFn_t output_fn)
{
  CustomContext * const context = parseContext(customContext);

  return context->execute(numPayloads, payloads, input_fn, output_fn);
}


