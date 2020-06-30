/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef TENSORFLOW_TRAITS_H_
#define TENSORFLOW_TRAITS_H_

using namespace fastertransformer;
namespace tensorflow
{
  template <typename T> class TFTraits;
  
  template <>
  class TFTraits<float>
  {
    public:
      typedef float DataType;
      static const OperationType OpType = OperationType::FP32;
  };

  template <>
  class TFTraits<Eigen::half>
  {
    public:
      typedef half DataType;
      static const OperationType OpType = OperationType::FP16;
  };

} //namespace tensorflow
#endif
