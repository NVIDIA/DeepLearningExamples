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

#include <exception>
#include <iostream>

namespace
{

std::vector<UnitTest*>* s_tests = nullptr;
}

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

bool UnitTest::runAll()
{
  size_t numPassed = 0;
  size_t numTests = 0;

  if (s_tests) {
    numTests = s_tests->size();
    for (UnitTest* const test : *s_tests) {
      try {
        test->run();

        if (test->passed()) {
          std::cout << "Test: " << test->fullname() << " passed." << std::endl;
          ++numPassed;
          continue;
        }
      } catch (const TestException&) {
        // assertion failed
      } catch (const std::exception& e) {
        std::cout << "Unhandled excpetion: " << e.what() << std::endl;
      }
      std::cout << "Test: " << test->fullname() << " failed." << std::endl;
    }
  }

  std::cout << numPassed << " / " << numTests << " passed." << std::endl;

  return numPassed == numTests;
}

void UnitTest::registerTest(UnitTest* const test)
{
  if (!s_tests) {
    s_tests = new std::vector<UnitTest*>(0);
  }

  s_tests->emplace_back(test);
}

/******************************************************************************
 * CONSTRUCTORS / DESTRUCTOR **************************************************
 *****************************************************************************/

UnitTest::UnitTest(const std::string& filename, const std::string& name)
    : m_nullStream(), m_passed(true), m_filename(filename), m_name(name)
{
  registerTest(this);
}

/******************************************************************************
 * PUBLIC METHODS *************************************************************
 *****************************************************************************/

std::string UnitTest::fullname() const
{
  return m_filename + "__" + m_name;
}

bool UnitTest::passed() const
{
  return m_passed;
}

/******************************************************************************
 * PROTECTED METHODS **********************************************************
 *****************************************************************************/

void UnitTest::failure()
{
  m_passed = false;
}

/******************************************************************************
 * MAIN ***********************************************************************
 *****************************************************************************/

int main(int /*argc*/, char** /*argv*/)
{
  if (UnitTest::runAll()) {
    return 0;
  } else {
    return 1;
  }
}
