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

#ifndef TT2I_UTILS_H
#define TT2I_UTILS_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts
{

class Utils
{
public:
    /**
     * @brief Convert a string to lower-case.
     *
     * @param str The string.
     *
     * @return The lower-case version of the string.
     */
    static std::string toLower(const std::string& str)
    {
        std::string lower(str);
        for (char& c : lower)
        {
            c = std::tolower(c);
        }

        return lower;
    }

    /**
     * @brief Check if a given filename ends with a given extension.
     *
     * @param str The filename.
     * @param ext The extension.
     *
     * @return True if the filename ends with the given extension.
     */
    static bool hasExtension(const std::string& str, const std::string& ext)
    {
        return str.length() >= ext.length()
            && std::equal(str.begin() + (str.length() - ext.length()), str.end(), ext.begin());
    }

    /**
     * @brief Convert a string to a bool value. It accepts "y", "yes", "true",
     * and "1", ignoring capitalization, as true. It accepts "n", "no",
     * "false", and "0", ignoring capitalization, as false. Otherwise an
     * exception is thrown.
     *
     * @param str The string to parse.
     *
     * @return True or false depending on the value of the string.
     */
    static bool parseBool(const std::string& str)
    {
        const std::string lower = toLower(str);
        if (lower == "y" || lower == "yes" || lower == "true" || lower == "1")
        {
            return true;
        }
        else if (lower == "n" || lower == "no" || lower == "false" || lower == "0")
        {
            return false;
        }
        else
        {
            throw std::runtime_error("Unable to parse bool from '" + str + "'.");
        }
    }

    /**
     * @brief Evaluate the 'sigmoid' function: f(x) = 1 / (1 + e^{-x}).
     *
     * @param x The value to evaluate the sigmoid function at.
     *
     * @return The result.
     */
    static float sigmoid(const float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

} // namespace tts

#endif
