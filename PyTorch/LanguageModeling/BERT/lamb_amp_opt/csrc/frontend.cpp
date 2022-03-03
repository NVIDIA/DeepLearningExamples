#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

void multi_tensor_lamb_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const at::Tensor step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  at::Tensor global_grad_norm,
  at::Tensor max_grad_norm,
  at::optional<bool> use_nvlamb_python,
  at::Tensor found_inf,
  at::Tensor inv_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_lamb", &multi_tensor_lamb_cuda,
        "Computes and apply update for LAMB optimizer");
}
