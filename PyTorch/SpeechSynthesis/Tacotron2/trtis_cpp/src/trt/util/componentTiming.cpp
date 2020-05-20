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

#include "componentTiming.h"

#include <stdexcept>

namespace tts
{

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

ComponentTiming::ComponentTiming(const std::string& name, const double duration)
    : mName(name)
    , mDuration(duration)
    , mSubTimings()
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void ComponentTiming::addSubTiming(const ComponentTiming& timing)
{
    mSubTimings.emplace_back(timing);
}

void ComponentTiming::addSubTiming(const std::string& name, const double duration)
{
    mSubTimings.emplace_back(name, duration);
}

void ComponentTiming::print(std::ostream& stream, const int numRuns) const
{
    output(0, stream, numRuns);
}

std::string ComponentTiming::getName() const
{
    return mName;
}

double ComponentTiming::getDuration() const
{
    return mDuration;
}

ComponentTiming ComponentTiming::getSubTiming(const std::string& name) const
{
    for (const ComponentTiming& timing : mSubTimings)
    {
        if (timing.getName() == name)
        {
            return timing;
        }
    }
    throw std::runtime_error("Unable to find timing named '" + name + "'.");
}

/******************************************************************************
 * PRIVATE METHODS ************************************************************
 *****************************************************************************/

void ComponentTiming::output(const int level, std::ostream& stream, const int numRuns, const double parentTime) const
{
    for (int i = 0; i < level; ++i)
    {
        stream << "  ";
    }

    if (numRuns == 0)
    {
        throw std::runtime_error("Cannot compute average time of 0 runs.");
    }

    stream << mName << ": " << (mDuration / numRuns) << " s";

    if (level > 0 && parentTime > 0.0)
    {
        stream << " (% " << 100.0 * (mDuration / parentTime) << ")";
    }
    stream << std::endl;

    for (const ComponentTiming& t : mSubTimings)
    {
        if (t.getDuration() > 0)
        {
            t.output(level + 1, stream, numRuns, mDuration);
        }
    }
}

} // namespace tts
