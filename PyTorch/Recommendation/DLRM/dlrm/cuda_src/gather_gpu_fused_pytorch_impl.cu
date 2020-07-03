#include <torch/extension.h>
#include <torch/types.h>
#include <stdexcept>
#include "gather_gpu_fused.cu"

// plugin functions instantiated to do only mixed-precision execution
torch::Tensor gatherGPUFusedFwdTorch(torch::Tensor embedding, torch::Tensor indices, torch::Tensor offsets,
                                        bool amp_train) {
  auto size = indices.sizes();
  auto batch_size = size[0];
  auto num_features = size[1];

  size = embedding.sizes();
  auto embedding_vector_dim = size[1];
  auto embedding_table_rows = size[0];    // not really need this

//   if (embedding.scalar_type() != torch::ScalarType::Float) {
//     throw std::invalid_argument("Invalid input type.");
//   }

  int64_t outputShape[3] = {batch_size, num_features, embedding_vector_dim};
  torch::Tensor output;

  if (embedding.scalar_type() == torch::ScalarType::Float) {
    if (amp_train) {
        output = torch::empty(c10::IntArrayRef(outputShape), embedding.options().dtype(torch::ScalarType::Half));
        gather_gpu_fused_fwd(embedding.contiguous().data_ptr<float>(),
                                offsets.contiguous().data_ptr<int64_t>(),
                                indices.contiguous().data_ptr<int64_t>(),
                                output.contiguous().data_ptr<at::Half>(),
                                batch_size);
    }
    else {
        output = torch::empty(c10::IntArrayRef(outputShape), embedding.options().dtype(torch::ScalarType::Float));
        gather_gpu_fused_fwd(embedding.contiguous().data_ptr<float>(),
                                offsets.contiguous().data_ptr<int64_t>(),
                                indices.contiguous().data_ptr<int64_t>(),
                                output.contiguous().data_ptr<float>(),
                                batch_size);
    }
  }
  else {
    output = torch::empty(c10::IntArrayRef(outputShape), embedding.options().dtype(torch::ScalarType::Half));
    gather_gpu_fused_fwd(embedding.contiguous().data_ptr<at::Half>(),
                            offsets.contiguous().data_ptr<int64_t>(),
                            indices.contiguous().data_ptr<int64_t>(),
                            output.contiguous().data_ptr<at::Half>(),
                            batch_size);
  }
  return output;
}

torch::Tensor gatherGPUFusedBwdTorch(torch::Tensor embedding, torch::Tensor indices,
                                          torch::Tensor offsets, torch::Tensor upstreamGrad) {
  if (embedding.scalar_type() != torch::ScalarType::Float) {
    throw std::invalid_argument("Invalid input type.");
  }

  auto size = upstreamGrad.sizes();
  auto batch_size = size[0];
  auto num_features = size[1];
  auto embedding_vector_dim = size[2];

  size = indices.sizes();
  auto sparse_tensor_indices_dim = size[0] * size[1];
  int64_t indices_outputShape[2] = {1, sparse_tensor_indices_dim};

  auto sparse_tensor_values_0 = batch_size * num_features;
  auto sparse_tensor_values_1 = embedding_vector_dim;
  int64_t values_outputShape[2] = {sparse_tensor_values_0, sparse_tensor_values_1};

  auto sparse_grad_indices_tensor = torch::empty(c10::IntArrayRef(indices_outputShape), indices.options());

  auto sparse_grad_values_tensor = torch::empty(c10::IntArrayRef(values_outputShape),
                                                  upstreamGrad.options().dtype(torch::ScalarType::Float));

  // this is the shape of output gradient vector
  int64_t sparse_tensor_shape[2] = {embedding.sizes()[0], embedding_vector_dim};

  if (upstreamGrad.scalar_type() == torch::ScalarType::Half) {
    gather_gpu_fused_bwd(upstreamGrad.contiguous().data_ptr<at::Half>(),
                            indices.contiguous().data_ptr<int64_t>(),
                            offsets.contiguous().data_ptr<int64_t>(),
                            sparse_grad_values_tensor.contiguous().data_ptr<float>(),
                            sparse_grad_indices_tensor.contiguous().data_ptr<int64_t>(),
                        (int)batch_size, (int)num_features, (int)embedding_vector_dim);
  }
  else {
    gather_gpu_fused_bwd(upstreamGrad.contiguous().data_ptr<float>(),
                            indices.contiguous().data_ptr<int64_t>(),
                            offsets.contiguous().data_ptr<int64_t>(),
                            sparse_grad_values_tensor.contiguous().data_ptr<float>(),
                            sparse_grad_indices_tensor.contiguous().data_ptr<int64_t>(),
                        (int)batch_size, (int)num_features, (int)embedding_vector_dim);
  }

  return torch::_sparse_coo_tensor_with_dims_and_tensors(1, 1, c10::IntArrayRef(sparse_tensor_shape),
                                                    sparse_grad_indices_tensor, sparse_grad_values_tensor,
                                                    sparse_grad_values_tensor.options().layout(c10::Layout::Sparse));
}
