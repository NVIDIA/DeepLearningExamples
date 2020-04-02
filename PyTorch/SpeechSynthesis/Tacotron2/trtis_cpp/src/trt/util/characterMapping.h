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

#ifndef TT2I_CHARACTERMAPPING_H
#define TT2I_CHARACTERMAPPING_H

#include "timedObject.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace tts
{

class CharacterMapping : public TimedObject
{
public:
    /**
     * @brief Create a default character mapping. This maps standard 'ascii'
     * characters as follows:
     * ```
     * _-!\'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
     * ```
     *
     * @return The character mapping.
     */
    static CharacterMapping defaultMapping();

    /**
     * @brief Create an empty character mapping.
     */
    CharacterMapping();

    /**
     * @brief Create a character mapping based on the given set of characters.
     *
     * @param mapping The set of characters to map in order.
     */
    CharacterMapping(const std::vector<char>& mapping);

    /**
     * @brief Create a character mapping based on the given set of string,
     * treating each string as a symbol.
     *
     * @param mapping The set of symbols to map in order.
     */
    CharacterMapping(const std::vector<std::string>& mapping);

    /**
     * @brief Set the given character to the given sequence number.
     *
     * @param c The character.
     * @param n The sequence number.
     */
    void set(char c, int32_t n);

    /**
     * @brief Set the given symbol to the given sequence number.
     *
     * @param c The symbol.
     * @param n The sequence number.
     */
    void set(const std::string& c, int32_t n);

    /**
     * @brief Map the given character to the given sequence number.
     *
     * @param c The character to map.
     *
     * @return The sequence number.
     */
    int32_t get(char c) const;

    /**
     * @brief Map the given symbol to the given sequence number.
     *
     * @param c The symbol to map.
     *
     * @return The sequence number.
     */
    int32_t get(const std::string& c) const;

    /**
     * @brief Convert a string of symbols to a sequence.
     *
     * @param input The string of symbols.
     *
     * @return The sequence.
     */
    std::vector<int32_t> map(const std::string& input);

    /**
     * @brief Map a set of bytes to sequence numbers. Several bytes may map to a
     * single sequence nuumber, thus the output length will be equal or less than
     * the input length.
     *
     * @param input The input.
     * @param inputSize The length of the input in characters.
     * @param output Must be of size at least `inputSize`.
     * @param outputSize The final size of the output (will be equal to or less
     * than `inputSize`).
     */
    void map(const char* input, size_t inputSize, int32_t* output, size_t* outputSize);

private:
    std::unordered_map<std::string, int32_t> mMapping;
};

} // namespace tts

#endif
