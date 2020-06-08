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

#include "UnitTest.hpp"
#include "characterMapping.h"

using namespace tts;

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST(MapAsciiTest)
{
  const std::string text(
    "printing, in the only sense with which we are at present concerned, differs "
    "from most if not from all the arts and crafts represented in the exhibition in "
    "being comparatively modern.");

  CharacterMapping cm = CharacterMapping::defaultMapping();

  const std::vector<int32_t> sequence = cm.map(text);

  const std::vector<int32_t> expSequence{
53, 55, 46, 51, 57, 46, 51, 44, 6 , 11, 46, 51, 11, 57, 45, 42, 11, 52, 51, 49,
62, 11, 56, 42, 51, 56, 42, 11, 60, 46, 57, 45, 11, 60, 45, 46, 40, 45, 11, 60,
42, 11, 38, 55, 42, 11, 38, 57, 11, 53, 55, 42, 56, 42, 51, 57, 11, 40, 52, 51,
40, 42, 55, 51, 42, 41, 6, 11, 41, 46, 43, 43, 42, 55, 56, 11, 43, 55, 52, 50,
11, 50, 52, 56, 57, 11, 46, 43, 11, 51, 52, 57, 11, 43, 55, 52, 50, 11, 38, 49,
49, 11, 57, 45, 42, 11, 38, 55, 57, 56, 11, 38, 51, 41, 11, 40, 55, 38, 43, 57,
56, 11, 55, 42, 53, 55, 42, 56, 42, 51, 57, 42, 41, 11, 46, 51, 11, 57, 45, 42,
11, 42, 61, 45, 46, 39, 46, 57, 46, 52, 51, 11, 46, 51, 11, 39, 42, 46, 51, 44,
11, 40, 52, 50, 53, 38, 55, 38, 57, 46, 59, 42, 49, 62, 11, 50, 52, 41, 42, 55,
51, 7 };

  ASSERT_EQ(sequence.size(), expSequence.size());
  for (size_t i = 0; i < expSequence.size(); ++i) {
    EXPECT_EQ(expSequence[i], sequence[i]);
  }
}

TEST(MapArpabetTest)
{
  const std::string text("Hello {@AE0}ther {@UW}{@AO}rld.");

  CharacterMapping cm = CharacterMapping::defaultMapping();

  const std::vector<int32_t> sequence = cm.map(text);

  const std::vector<int32_t> expSequence{
      45, 42, 49, 49, 52, 11, 69, 57, 45, 42, 55, 11, 139, 76, 55, 49, 41, 7};

  ASSERT_EQ(sequence.size(), expSequence.size());
  for (size_t i = 0; i < expSequence.size(); ++i) {
    EXPECT_EQ(expSequence[i], sequence[i]);
  }
}
