// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"

#include "volta/dot_based_interact_volta.h"
#include "ampere/dot_based_interact_ampere.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::half half;

namespace functor {

template <typename Device, typename T>
struct DotBasedInteractGradFunctor {
  void operator()(const Device& d, const T* input, const T* upstream_grad, T* grad,
                  T* bottom_mlp_grad, int64 batch_size, int64 num_rows, int64 num_cols);
};

template <>
struct DotBasedInteractGradFunctor<GPUDevice, float> {
  void operator()(const GPUDevice& d, const float* input, const float* upstream_grad, float* grad,
                  float* bottom_mlp_grad, int64 batch_size, int64 num_rows, int64 num_cols) {
    int major = d.majorDeviceVersion();
    if (major >= 8) {
      dotBasedInteractAmpereTF32Bwd(input,
                                    upstream_grad,
                                    grad,
                                    bottom_mlp_grad,
                                    batch_size,
                                    num_rows,
                                    num_cols,
                                    d.stream());
    } else if (major == 7) {
      dotBasedInteractVoltaF32Bwd(input,
                                  upstream_grad,
                                  grad,
                                  bottom_mlp_grad,
                                  batch_size,
                                  num_rows,
                                  num_cols,
                                  d.stream());
    }
  }
};

template <>
struct DotBasedInteractGradFunctor<GPUDevice, half> {
  void operator()(const GPUDevice& d, const half* input, const half* upstream_grad, half* grad,
                  half* bottom_mlp_grad, int64 batch_size, int64 num_rows, int64 num_cols) {
    int major = d.majorDeviceVersion();
    if (major >= 8) {
      dotBasedInteractAmpereF16Bwd(input,
                                   upstream_grad,
                                   grad,
                                   bottom_mlp_grad,
                                   batch_size,
                                   num_rows,
                                   num_cols,
                                   d.stream());
    } else if (major == 7) {
      dotBasedInteractVoltaF16Bwd(input,
                                  upstream_grad,
                                  grad,
                                  bottom_mlp_grad,
                                  batch_size,
                                  num_rows,
                                  num_cols,
                                  d.stream());
    }
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class DotBasedInteractGradOp : public OpKernel {
 public:
  explicit DotBasedInteractGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Grab the bottom_mlp_output tensor
    const Tensor& upstream_grad_tensor = context->input(1);

    // Calculate the output tensor shape
    TensorShape input_shape = input_tensor.shape();
    int64 batch_size = input_shape.dim_size(0);
    int64 num_rows = input_shape.dim_size(1);
    int64 num_cols = input_shape.dim_size(2);
    TensorShape bottom_mlp_grad_shape({batch_size, num_cols});

    // Create the grad output tensor
    Tensor* grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape,
                                                     &grad_tensor));

    // Create the bottom mlp grad output tensor
    Tensor* bottom_mlp_grad_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, bottom_mlp_grad_shape,
                                                     &bottom_mlp_grad_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    // GPU Architecture
    GPUDevice device = ((GPUDevice) context->eigen_device<Device>());
    OP_REQUIRES(context, device.majorDeviceVersion() >= 7,
                errors::InvalidArgument("GPU not supported (need Volta or higher)"));

    DotBasedInteractGradFunctor<Device, T>()(
        device,
        input_tensor.flat<T>().data(),
        upstream_grad_tensor.flat<T>().data(),
        grad_tensor->flat<T>().data(),
        bottom_mlp_grad_tensor->flat<T>().data(),
        batch_size,
        num_rows,
        num_cols);
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DotBasedInteractGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DotBasedInteractGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(half);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
