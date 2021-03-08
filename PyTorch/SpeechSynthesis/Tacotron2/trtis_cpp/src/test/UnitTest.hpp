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

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TT2I_UNITTEST_HPP
#define TT2I_UNITTEST_HPP

#define _TEST(test_name, name)                                                 \
  class test_name : public UnitTest                                            \
  {                                                                            \
  public:                                                                      \
    test_name() : UnitTest(__FILE__, #name){};                                 \
    void run() override;                                                       \
  };                                                                           \
  test_name test_name##_instance;                                              \
  void test_name::run()

#define TEST(name) _TEST(test_##__FILE__##__##name, name)

#define ASSERT_TRUE(x)                                                         \
  do {                                                                         \
    if (!(x)) {                                                                \
      std::cerr << "ASSERT_TRUE: " << #x << "(" << (x) << ") is false at "     \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw TestException();                                                   \
    }                                                                          \
  } while (false)

#define ASSERT_EQ(x, y)                                                        \
  do {                                                                         \
    if (!((x) == (y))) {                                                       \
      std::cerr << "ASSERT_EQ: " << #x << "(" << (x) << ") != " << #y << "("   \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      throw TestException();                                                   \
    }                                                                          \
  } while (false)

#define ASSERT_LT(x, y)                                                        \
  do {                                                                         \
    if (!areComparable((x), (y)) || !((x) < (y))) {                            \
      std::cerr << "ASSERT_LT: " << #x << "(" << (x) << ") !< " << #y << "("   \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      throw TestException();                                                   \
    }                                                                          \
  } while (false)

#define ASSERT_LE(x, y)                                                        \
  do {                                                                         \
    if (!areComparable((x), (y)) || !((x) <= (y))) {                           \
      std::cerr << "ASSERT_LE: " << #x << "(" << (x) << ") !<= " << #y << "("  \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      throw TestException();                                                   \
    }                                                                          \
  } while (false)

#define ASSERT_GT(x, y)                                                        \
  do {                                                                         \
    if (!areComparable((x), (y)) || !((x) > (y))) {                            \
      std::cerr << "ASSERT_GT: " << #x << "(" << (x) << ") !> " << #y << "("   \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      throw TestException();                                                   \
    }                                                                          \
  } while (false)

#define ASSERT_GE(x, y)                                                        \
  do {                                                                         \
    if (!areComparable((x), (y)) || !((x) >= (y))) {                           \
      std::cerr << "ASSERT_GE: " << #x << "(" << (x) << ") !>= " << #y << "("  \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      throw TestException();                                                   \
    }                                                                          \
  } while (false)

#define EXPECT_TRUE(x)                                                         \
  [&]() {                                                                      \
    if (!(x)) {                                                                \
      std::cerr << "EXPECT_TRUE: " << #x << "(" << (x) << ") is false at "     \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

#define EXPECT_EQ(x, y)                                                        \
  [&]() {                                                                      \
    if (!areComparable((x), (y)) || !((x) == (y))) {                           \
      std::cerr << "EXPECT_EQ: " << #x << "(" << (x) << ") != " << #y << "("   \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

#define EXPECT_LT(x, y)                                                        \
  [&]() {                                                                      \
    if (!areComparable((x), (y)) || !((x) < (y))) {                            \
      std::cerr << "EXPECT_LT: " << #x << "(" << (x) << ") !< " << #y << "("   \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

#define EXPECT_LE(x, y)                                                        \
  [&]() {                                                                      \
    if (!areComparable((x), (y)) || !((x) <= (y))) {                           \
      std::cerr << "EXPECT_LE: " << #x << "(" << (x) << ") !<= " << #y << "("  \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

#define EXPECT_GT(x, y)                                                        \
  [&]() {                                                                      \
    if (!areComparable((x), (y)) || !((x) > (y))) {                            \
      std::cerr << "EXPECT_GT: " << #x << "(" << (x) << ") !> " << #y << "("   \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

#define EXPECT_GE(x, y)                                                        \
  [&]() {                                                                      \
    if (!areComparable((x), (y)) || !((x) >= (y))) {                           \
      std::cerr << "EXPECT_GE: " << #x << "(" << (x) << ") !>= " << #y << "("  \
                << (y) << ") "                                                 \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

#define EXPECT_NEAR(x, y, t)                                                   \
  [&]() {                                                                      \
    auto diff = std::abs((x) - (y));                                           \
    if (!areComparable((x), (y)) || diff > (t)) {                              \
      std::cerr << "EXPECT_NEAR: " << #x << "(" << (x) << ") !~= " << #y       \
                << "(" << (y) << ") "                                          \
                << " within (" << diff << "/" #t << ") at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                      \
      this->failure();                                                         \
      return CheckOutput(true);                                                \
    } else {                                                                   \
      return CheckOutput(false);                                               \
    }                                                                          \
  }()

class CheckOutput
{
public:
  CheckOutput(bool output) : m_displayOutput(output), m_output()
  {
  }

  CheckOutput(CheckOutput&& other)
      : m_displayOutput(other.m_displayOutput),
        m_output(std::move(other.m_output))
  {
    other.m_displayOutput = false;
  }

  ~CheckOutput()
  {
    if (m_displayOutput && !m_output.str().empty()) {
      std::cerr << m_output.str() << std::endl;
    }
  }

  template <typename T>
  CheckOutput& operator<<(const T& obj)
  {
    m_output << obj;

    return *this;
  }

private:
  bool m_displayOutput;
  std::ostringstream m_output;
};

class TestException : public std::runtime_error
{
public:
  TestException() : std::runtime_error("TestFailed"){};
};

class UnitTest
{
public:
  static bool runAll();

  static void registerTest(UnitTest* test);

  UnitTest(const std::string& filename, const std::string& name);

  virtual ~UnitTest() = default;

  virtual void run() = 0;

  std::string fullname() const;

  bool passed() const;

protected:
  void failure();

  template <
      typename T,
      typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
  bool areComparable(T x, T y) const
  {
    return !std::isnan(x) && !std::isnan(y) &&
           (!std::isinf(x) || !std::isinf(y));
  }

  template <
      typename T,
      typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
  bool areComparable(T, T) const
  {
    return true;
  }

  std::ostringstream m_nullStream;

private:
  bool m_passed;
  std::string m_filename;
  std::string m_name;
};

#endif
