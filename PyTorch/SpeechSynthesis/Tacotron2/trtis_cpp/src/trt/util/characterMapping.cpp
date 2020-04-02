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

#include "characterMapping.h"

#include <cassert>
#include <stdexcept>

namespace tts
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr const char MULTI_CHAR_START = '{';
constexpr const char MULTI_CHAR_END = '}';

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{
std::string mkString(const char c)
{
    return std::string(&c, 1);
}

std::vector<std::string> toStrings(const std::vector<char>& vec)
{
    std::vector<std::string> outVec;
    for (const char c : vec)
    {
        outVec.emplace_back(mkString(c));
    }

    return outVec;
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

CharacterMapping CharacterMapping::defaultMapping()
{
    const std::string characters("_-!\'(),.:;? ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    const std::vector<std::string> arpaBet{"@AA", "@AA0", "@AA1", "@AA2", "@AE", "@AE0", "@AE1", "@AE2", "@AH", "@AH0",
        "@AH1", "@AH2", "@AO", "@AO0", "@AO1", "@AO2", "@AW", "@AW0", "@AW1", "@AW2", "@AY", "@AY0", "@AY1", "@AY2",
        "@B", "@CH", "@D", "@DH", "@EH", "@EH0", "@EH1", "@EH2", "@ER", "@ER0", "@ER1", "@ER2", "@EY", "@EY0", "@EY1",
        "@EY2", "@F", "@G", "@HH", "@IH", "@IH0", "@IH1", "@IH2", "@IY", "@IY0", "@IY1", "@IY2", "@JH", "@K", "@L",
        "@M", "@N", "@NG", "@OW", "@OW0", "@OW1", "@OW2", "@OY", "@OY0", "@OY1", "@OY2", "@P", "@R", "@S", "@SH", "@T",
        "@TH", "@UH", "@UH0", "@UH1", "@UH2", "@UW", "@UW0", "@UW1", "@UW2", "@V", "@W", "@Y", "@Z", "@ZH"};

    std::vector<std::string> totalChars;
    for (const char c : characters)
    {
        totalChars.emplace_back(mkString(c));
    }
    for (const std::string& c : arpaBet)
    {
        totalChars.emplace_back(c);
    }

    CharacterMapping mapping(totalChars);

    // the default mapping maps all upper case characters to lower case.
    for (const char c : std::string("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    {
        mapping.set(c, mapping.get(std::tolower(c)));
    }

    return mapping;
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

CharacterMapping::CharacterMapping()
    : TimedObject("CharacterMapping::map()")
    , mMapping()
{
    // do nothing
}

CharacterMapping::CharacterMapping(const std::vector<std::string>& mapping)
    : CharacterMapping()
{
    mMapping.reserve(1024);
    for (size_t i = 0; i < mapping.size(); ++i)
    {
        mMapping.emplace(mapping[i], i);
    }
}

CharacterMapping::CharacterMapping(const std::vector<char>& mapping)
    : CharacterMapping(toStrings(mapping))
{
    // do nothing
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

void CharacterMapping::set(const char c, const int32_t n)
{
    set(mkString(c), n);
}

void CharacterMapping::set(const std::string& c, const int32_t n)
{
    std::pair<std::unordered_map<std::string, int32_t>::iterator, bool> result = mMapping.emplace(c, n);
    if (!result.second)
    {
        result.first->second = n;
    }
}

int32_t CharacterMapping::get(const char c) const
{
    return get(mkString(c));
}

int32_t CharacterMapping::get(const std::string& c) const
{
    const std::unordered_map<std::string, int32_t>::const_iterator iter = mMapping.find(c);
    if (iter == mMapping.end())
    {
        if (c.empty())
        {
            throw std::runtime_error("Cannot map empty symbol.");
        }
        else if (c.length() == 1)
        {
            std::string charRep;
            const int sym = static_cast<uint8_t>(c[0]);
            if (std::isprint(sym))
            {
                charRep = "'" + c + "' (" + std::to_string(sym) + ")";
            }
            else
            {
                charRep = "(" + std::to_string(sym) + ")";
            }
            throw std::runtime_error("Could not find '" + charRep + "' in the alphabet.");
        }
        else
        {
            throw std::runtime_error("Could not find '" + c + "' in the alphabet.");
        }
    }
    else
    {
        return iter->second;
    }
}

std::vector<int32_t> CharacterMapping::map(const std::string& input)
{
    std::vector<int32_t> nums(input.size());
    size_t numSize;
    map(input.c_str(), input.size(), nums.data(), &numSize);
    nums.resize(numSize);

    return nums;
}

void CharacterMapping::map(
    const char* const input, const size_t inputSize, int32_t* const output, size_t* const outputSize)
{
    assert(input != nullptr);
    assert(output != nullptr);
    assert(outputSize != nullptr);

    try
    {
        startTiming();
        size_t pos = 0;

        std::string buffer;
        bool multiChar = false;
        for (size_t i = 0; i < inputSize; ++i)
        {
            assert(pos <= i);

            const char c = input[i];
            if (c == MULTI_CHAR_START)
            {
                if (multiChar)
                {
                    // error
                    throw std::runtime_error(
                        "Got two multi-byte character start symbols: '" + std::string(input, inputSize) + "'.");
                }

                multiChar = true;
                buffer = "";
            }
            else if (c == MULTI_CHAR_END)
            {
                if (!multiChar)
                {
                    // error
                    throw std::runtime_error(
                        "Got two multi-byte character end symbols: '" + std::string(input, inputSize) + "'.");
                }

                multiChar = false;
                output[pos++] = get(buffer);
            }
            else if (multiChar)
            {
                buffer += c;
            }
            else
            {
                output[pos++] = get(c);
            }
        }
        if (multiChar)
        {
            // error
            throw std::runtime_error("Ended in a multi-byte character.");
        }

        *outputSize = pos;
    }
    catch (...)
    {
        // don't leave the timer running
        stopTiming();
        throw;
    }
    stopTiming();
}

} // namespace tts
