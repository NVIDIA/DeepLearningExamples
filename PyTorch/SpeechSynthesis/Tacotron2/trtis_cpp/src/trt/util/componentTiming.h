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

#ifndef TT2I_COMPONENTTIMINGS_H
#define TT2I_COMPONENTTIMINGS_H

#include <ostream>
#include <string>
#include <vector>

namespace tts
{

class ComponentTiming
{
public:
    /**
     * @brief Create a new component timing object.
     *
     * @param name The name of the component.
     * @param duration The time duration in seconds.
     */
    ComponentTiming(const std::string& name, double duration);

    /**
     * @brief Add a timed sub-component.
     *
     * @param timing The timing of the sub-component.
     */
    void addSubTiming(const ComponentTiming& timing);

    /**
     * @brief Add a timed sub-component.
     *
     * @param name The name of the sub-component.
     * @param duration The time duration in seconds.
     */
    void addSubTiming(const std::string& name, double duration);

    /**
     * @brief Print out the time taken averaged over a given number of runs.
     *
     * @param stream The stream to print to.
     * @param numRuns The number of runs to average over.
     */
    void print(std::ostream& stream, int numRuns) const;

    /**
     * @brief Get the name of this component.
     *
     * @return The name.
     */
    std::string getName() const;

    /**
     * @brief Get the duration of this component in seconds.
     *
     * @return The duration.
     */
    double getDuration() const;

    /**
     * @brief Get the timing of a sub-component.
     *
     * @param name The name of the sub-component.
     *
     * @return The timing of the sub-component.
     */
    ComponentTiming getSubTiming(const std::string& name) const;

private:
    std::string mName;
    double mDuration;
    std::vector<ComponentTiming> mSubTimings;

    /**
     * @brief Output the timing to stream. This will write a line like:
     * ```
     *     ComponentName: 7.32s (54.3%)
     * ```
     *
     * @param level The level of indentation.
     * @param stream The stream to write to.
     * @param numRuns The number of runs to average.
     * @param parentTime The total time taken by the parent.
     */
    void output(int level, std::ostream& stream, int numRuns, double parentTime = 0.0) const;
};

} // namespace tts

#endif
