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

#include "CharacterMappingReader.hpp"

#include <fstream>
#include <stdexcept>

namespace tts
{

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

bool isBlank(const std::string& line)
{
  for (const unsigned char c : line) {
    if (!std::isspace(c)) {
      return false;
    }
  }
  return true;
}

bool isComment(const std::string& line)
{
  for (const unsigned char c : line) {
    if (std::isspace(c)) {
      // keep searching
    } else if (c == '#') {
      return true;
    }
  }
  return false;
}

void parseKeyPair(
    const std::string& line, int* const num, std::string* const symbol)
{
  assert(num != nullptr);
  assert(symbol != nullptr);
  for (size_t i = 1; i + 1 < line.size(); ++i) {
    if (std::isspace(static_cast<unsigned char>(line[i]))) {
      // a valid key pair will be a number, a whitespace, and the rest will be
      // treated as the symbol.
      *num = std::stol(line.substr(0, i));
      *symbol = line.substr(i + 1);
      return;
    }
  }

  // if we found no space
  throw std::runtime_error("Failed to parse line '" + line + "'.");
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

CharacterMapping
CharacterMappingReader::loadFromFile(const std::string& filename)
{
  std::ifstream fin(filename);
  if (!fin.good()) {
    throw std::runtime_error("Failed to open '" + filename + "'.");
  }

  // read the file line by line
  CharacterMapping mapping;
  std::string line;
  std::string symbol;
  int num;
  while (std::getline(fin, line)) {
    if (isBlank(line)) {
      // do nothing
    } else if (isComment(line)) {
      // do nothing
    } else {
      parseKeyPair(line, &num, &symbol);

      mapping.set(symbol, num);
    }
  }
  if (fin.bad()) {
    throw std::runtime_error("Error while reading '" + filename + "'.");
  }

  return mapping;
}

} // namespace tts
