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

#ifndef TT2I_TIMEDOBJECT_H
#define TT2I_TIMEDOBJECT_H

#include "componentTiming.h"

#include "timer.h"

#include <string>

namespace tts
{

class TimedObject
{
public:
    /**
     * @brief Create a new timed object.
     *
     * @param name The name of the object.
     */
    TimedObject(const std::string& name)
        : mName(name)
        , mTimer()
        , mChildren()
    {
        // do nothing
    }

    /**
     * @brief Virtual destructor.
     */
    virtual ~TimedObject() = default;

    /**
     * @brief Get the timing of this current object (and all of its children).
     *
     * @return The timing.
     */
    virtual ComponentTiming getTiming() const
    {
      ComponentTiming time(mName, mTimer.poll());

      for (const TimedObject* const child : mChildren) {
        time.addSubTiming(child->getTiming());
        }

        return time;
    }

    /**
     * @brief Reset the timing of the current object (and all of its children).
     */
    void resetTiming()
    {
        mTimer.reset();

        for (TimedObject* const child : mChildren)
        {
            child->resetTiming();
        }
    }

    /**
     * @brief Print the timing of the current object (and all of its children)
     * to the given stream.
     *
     * @param stream The stream to print to.
     * @param numRuns The number of runs to average the times over.
     */
    void printTiming(std::ostream& stream, const int numRuns = 1) const
    {
        getTiming().print(stream, numRuns);
    }

protected:
    /**
     * @brief Add a child object. The child object must remain at this memory
     * location until this object is destroyed.
     *
     * @param child The child object.
     */
    void addChild(TimedObject* const child)
    {
        mChildren.emplace_back(child);
    }

    /**
     * @brief Start the internal timer.
     */
    void startTiming()
    {
        mTimer.start();
    }

    /**
     * @brief Stop the internal timer.
     */
    void stopTiming()
    {
        mTimer.stop();
    }

private:
    std::string mName;
    Timer mTimer;

    std::vector<TimedObject*> mChildren;
};

} // namespace tts

#endif
