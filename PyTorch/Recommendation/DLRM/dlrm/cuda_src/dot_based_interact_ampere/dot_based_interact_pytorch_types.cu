#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>

#include "dot_based_interact.cu"
#include "dot_based_interact_fp32.cu"
#include "dot_based_interact_tf32.cu"


torch::Tensor dotBasedInteractFwdTorch(torch::Tensor input, torch::Tensor bottom_mlp_output) {
  const uint kPaddingSize = 1;
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];
  uint output_size = ((num_rows * (num_rows - 1)) >> 1) + num_cols + kPaddingSize;

  int64_t outputShape[2] = {batch_size, output_size};
  auto output = torch::empty(c10::IntArrayRef(outputShape), input.options());
  if (input.scalar_type() == torch::ScalarType::Half && bottom_mlp_output.scalar_type() == torch::ScalarType::Half) {
    dotBasedInteractFwd(input.contiguous().data_ptr<at::Half>(),
                        bottom_mlp_output.contiguous().data_ptr<at::Half>(),
                        output.contiguous().data_ptr<at::Half>(),
                        batch_size,
                        num_rows,
                        num_cols);
  } else if (input.scalar_type() == torch::ScalarType::Float &&
             bottom_mlp_output.scalar_type() == torch::ScalarType::Float) {

    dotBasedInteractTF32Fwd(input.contiguous().data_ptr<float>(),
                            bottom_mlp_output.contiguous().data_ptr<float>(),
                            output.contiguous().data_ptr<float>(),
                            batch_size,
                            num_rows,
                            num_cols);
  } else {
    throw std::invalid_argument("Invalid input type.");
  }
  return output;
}

std::vector<torch::Tensor> dotBasedInteractBwdTorch(torch::Tensor input, torch::Tensor upstreamGrad) {
  auto size = input.sizes();
  auto batch_size = size[0];
  auto num_rows = size[1];
  auto num_cols = size[2];

  auto outputGrad = torch::empty_like(input);
  int64_t outputShape[2] = {batch_size, num_cols};
  auto mlp_grad = torch::empty(c10::IntArrayRef(outputShape), input.options());

  if (input.scalar_type() == torch::ScalarType::Half && upstreamGrad.scalar_type() == torch::ScalarType::Half) {
    dotBasedInteractBwd(input.contiguous().data_ptr<at::Half>(),
                        upstreamGrad.contiguous().data_ptr<at::Half>(),
                        outputGrad.contiguous().data_ptr<at::Half>(),
                        mlp_grad.contiguous().data_ptr<at::Half>(),
                        batch_size,
                        num_rows,
                        num_cols);
  } else if (input.scalar_type() == torch::ScalarType::Float &&
             upstreamGrad.scalar_type() == torch::ScalarType::Float) {

    dotBasedInteractTF32Bwd(input.contiguous().data_ptr<float>(),
                            upstreamGrad.contiguous().data_ptr<float>(),
                            outputGrad.contiguous().data_ptr<float>(),
                            mlp_grad.contiguous().data_ptr<float>(),
                            batch_size,
                            num_rows,
                            num_cols);
  } else {
    throw std::invalid_argument("Invalid input type.");
  }
  return {outputGrad, mlp_grad};
}
