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

#include "binding.h"

#include <cassert>
#include <stdexcept>
#include <string>

using namespace nvinfer1;

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr size_t MIN_SIZE = 64;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

Binding::Binding()
    : mBindings()
{
    mBindings.reserve(MIN_SIZE);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void** Binding::getBindings()
{
    return mBindings.data();
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void Binding::setVoidBinding(const ICudaEngine& engine, const char* const name, void* const ptr)
{
    const int pos = engine.getBindingIndex(name);
    if (pos < 0)
    {
        throw std::runtime_error("Invalid binding index " + std::to_string(pos) + " for '" + name + "'.");
    }
    if (pos + 1 > static_cast<int>(mBindings.size()))
    {
        mBindings.resize(pos + 1);
    }

    mBindings[pos] = ptr;
}

} // namespace tts
