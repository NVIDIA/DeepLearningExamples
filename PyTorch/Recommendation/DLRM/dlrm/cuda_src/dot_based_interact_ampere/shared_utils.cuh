#pragma once

#include <math.h>

#define CHK_CUDA(expression)                                                                                        \
  {                                                                                                                 \
    cudaError_t status = (expression);                                                                              \
    if (status != cudaSuccess) {                                                                                    \
      std::cerr << "Error in file: " << __FILE__ << ", on line: " << __LINE__ << ": " << cudaGetErrorString(status) \
                << std::endl;                                                                                       \
      std::exit(EXIT_FAILURE);                                                                                      \
    }                                                                                                               \
  }

template <uint x>
struct Log2 {
  static constexpr uint value = 1 + Log2<x / 2>::value;
};
template <>
struct Log2<1> {
  static constexpr uint value = 0;
};
