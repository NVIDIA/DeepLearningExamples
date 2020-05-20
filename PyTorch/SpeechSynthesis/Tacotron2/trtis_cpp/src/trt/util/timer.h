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
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TT2I_TIMER_H
#define TT2I_TIMER_H

#include <cassert>
#include <chrono>

namespace tts
{

class Timer
{
public:
  using Clock = std::chrono::high_resolution_clock;

  /**
   * @brief Create a new timer in a stopped state.
   */
  Timer() : m_total(0), m_start(), m_running(false)
  {
    // do nothing
  }

  /**
   * @brief Start the timer. Only stopped timers can be started.
   */
  void start()
  {
    assert(!m_running);
    m_running = true;
    m_start = std::chrono::high_resolution_clock::now();
  }

  /**
   * @brief Stop the timer. Only running timers can be stopped.
   */
  void stop()
  {
    assert(m_running);
    m_running = false;
    const auto elapsed = Clock::now() - m_start;
    const double seconds
        = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count()
          / 1000000.0;
    m_total += seconds;
  }

  /**
   * @brief Get the current duration of the timer. Only stopped timers can be
   * polled.
   *
   * @return
   */
  double poll() const
  {
    assert(!m_running);
    return m_total;
  }

  /**
   * @brief Reset the timer to zero. Running timers will be stopped.
   */
  void reset()
  {
    m_running = false;
    m_total = 0;
  }

private:
  double m_total;
  Clock::time_point m_start;
  bool m_running;
};

} // namespace tts

#endif
