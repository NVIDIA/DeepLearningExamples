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

#include "jsonModelImporter.h"

#include <fstream>

namespace tts
{

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

std::string mkString(const char c)
{
    return std::string(&c, 1);
}

char nextNonWhiteSpace(std::istream& stream)
{
    char c;
    do
    {
        stream.get(c);
        if (stream.fail())
        {
            throw std::runtime_error("Failed to read next char at position " + std::to_string(stream.tellg()) + ".");
        }
    } while (std::isspace(c));
    return c;
}

char peekNextNonWhiteSpace(std::istream& stream)
{
    char c;
    while (true)
    {
        c = stream.peek();
        if (stream.fail())
        {
            throw std::runtime_error("Failed to peek at next char at position " + std::to_string(stream.tellg()) + ".");
        }
        if (!std::isspace(c))
        {
            break;
        }
        else
        {
            // move past this white space character
            stream.get(c);
            if (stream.fail())
            {
                throw std::runtime_error(
                    "Failed to read next char at position " + std::to_string(stream.tellg()) + ".");
            }
        }
    }
    return c;
}

void expectNextCharacter(std::istream& stream, const char expected)
{
    const char c = nextNonWhiteSpace(stream);
    if (c != expected)
    {
        throw std::runtime_error("Failed to find '" + mkString(expected) + "' (found '" + mkString(c)
            + "' instead at position " + std::to_string(stream.tellg()) + ".");
    }
}

std::string readName(std::istream& stream)
{
    std::string name;

    expectNextCharacter(stream, '"');

    std::getline(stream, name, '"');
    return name;
}

float readNumber(std::istream& stream)
{
    float num;
    stream >> num;
    return num;
}

void readNextArray(std::istream& stream, std::vector<float>& data)
{
    const char c = peekNextNonWhiteSpace(stream);
    if (c == '[')
    {
        nextNonWhiteSpace(stream);
        // may be another array potentionally nested inside
        while (true)
        {
            char c = peekNextNonWhiteSpace(stream);
            if (c == '[')
            {
                // recurse
                readNextArray(stream, data);
            }
            else
            {
                // read actual array
                data.emplace_back(readNumber(stream));
            }
            // next character should be a ',' or a ']'
            c = nextNonWhiteSpace(stream);
            if (c == ']')
            {
                // end of array
                break;
            }
            else if (c != ',')
            {
                throw std::runtime_error(
                    "Invalid next character '" + mkString(c) + "' at position " + std::to_string(stream.tellg()) + ".");
            }
        }
    }
    else
    {
        data.emplace_back(readNumber(stream));
    }
}

std::vector<float> readTensor(std::istream& stream)
{
    std::vector<float> data;
    readNextArray(stream, data);

    return data;
}

} // namespace

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

JSONModelImporter::JSONModelImporter(const std::string& filename)
    : mWeights()
{
    std::ifstream fin(filename);
    if (!fin.good())
    {
        throw std::runtime_error("Failed to open '" + filename + "' for reading.");
    }

    char c;
    fin.get(c);
    if (c != '{')
    {
        throw std::runtime_error("First character must be '{', not " + mkString(c));
    }

    while (true)
    {
        // loop until we hit an error or the closing '}'
        const std::string name = readName(fin);

        expectNextCharacter(fin, ':');

        std::vector<float> tensor = readTensor(fin);

        // all but the last name in the path is the layer name
        const size_t layerNameEnd = name.find_last_of(".");
        std::string layerName = name.substr(0, layerNameEnd);
        const std::string dataName = name.substr(layerNameEnd + 1);

        // fix encoder names
        for (int i = 0; i < 3; ++i)
        {
            const std::string oldConvName = "encoder.convolutions." + std::to_string(i) + ".0.conv";
            if (layerName == oldConvName)
            {
                layerName = "encoder.convolutions." + std::to_string(i) + ".conv_layer.conv";
            }
            const std::string oldBatchName = "encoder.convolutions." + std::to_string(i) + ".1";
            if (layerName == oldBatchName)
            {
                layerName = "encoder.convolutions." + std::to_string(i) + ".batch_norm";
            }
        }

        // fix postnet names
        for (int i = 0; i < 5; ++i)
        {
            const std::string oldConvName = "postnet.convolutions." + std::to_string(i) + ".0.conv";
            if (layerName == oldConvName)
            {
                layerName = "postnet.convolutions." + std::to_string(i) + ".conv_layer.conv";
            }
            const std::string oldBatchName = "postnet.convolutions." + std::to_string(i) + ".1";
            if (layerName == oldBatchName)
            {
                layerName = "postnet.convolutions." + std::to_string(i) + ".batch_norm";
            }
        }

        auto iter = mWeights.find(layerName);
        if (iter == mWeights.end())
        {
            iter = mWeights.emplace(layerName, std::unique_ptr<LayerData>(new LayerData())).first;
        }
        else
        {
        }

        iter->second->add(dataName, tensor);

        if (peekNextNonWhiteSpace(fin) == '}')
        {
            break;
        }
        else
        {
            expectNextCharacter(fin, ',');
        }
    }
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

const LayerData* JSONModelImporter::getWeights(const std::vector<std::string>& path)
{
    std::string fullPath;
    for (size_t i = 0; i < path.size(); ++i)
    {
        fullPath += path[i];
        if (i + 1 < path.size())
        {
            fullPath += ".";
        }
    }

    auto iter = mWeights.find(fullPath);
    if (iter != mWeights.end())
    {
        return iter->second.get();
    }
    else
    {
        throw std::runtime_error("Unable to find '" + fullPath + "'");
    }
}

} // namespace tts
