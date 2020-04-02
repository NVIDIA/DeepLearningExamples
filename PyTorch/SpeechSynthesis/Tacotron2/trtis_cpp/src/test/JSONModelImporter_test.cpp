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
#include "jsonModelImporter.h"

#include <fstream>

using namespace tts;

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

TEST(ImportArraysTest)
{
  std::ofstream fout("test.json");

  fout << "{" << std::endl;
  fout << "\"test.layer.weight\" :" << std::endl;
  fout << "[[[1.0, 3.0, -5.0], [2.0, 1.0, 0.0]]]," << std::endl;
  fout << "\"test.layer.bias\" :" << std::endl;
  fout << "[[2.0, -3.0, 1.0]]" << std::endl;
  fout << "}" << std::endl;

  fout.flush();
  fout.close();

  JSONModelImporter importer("test.json");

  const LayerData * data = importer.getWeights({"test", "layer"});
  ASSERT_TRUE(data != nullptr);

  ASSERT_EQ(data->get("weight").count, 6);
  EXPECT_EQ(static_cast<const float*>(data->get("weight").values)[0], 1.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("weight").values)[1], 3.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("weight").values)[2], -5.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("weight").values)[3], 2.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("weight").values)[4], 1.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("weight").values)[5], 0.0f);

  ASSERT_EQ(data->get("bias").count, 3);
  EXPECT_EQ(static_cast<const float*>(data->get("bias").values)[0], 2.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("bias").values)[1], -3.0f);
  EXPECT_EQ(static_cast<const float*>(data->get("bias").values)[2], 1.0f);
}

TEST(ImportScalarTest)
{
  std::ofstream fout("test.json");

  fout << "{" << std::endl;
  fout << "\"test.layer.some_value\" :" << std::endl;
  fout << "3" << std::endl;
  fout << "}" << std::endl;

  fout.flush();
  fout.close();

  JSONModelImporter importer("test.json");

  const LayerData * data = importer.getWeights({"test", "layer"});
  ASSERT_TRUE(data != nullptr);

  ASSERT_EQ(data->get("some_value").count, 1);
  EXPECT_EQ(static_cast<const float*>(data->get("some_value").values)[0], 3.0f);
}
