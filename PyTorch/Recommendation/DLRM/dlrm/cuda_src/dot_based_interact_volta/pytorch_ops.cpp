#include <torch/extension.h>

torch::Tensor dotBasedInteractFwdTorch(torch::Tensor input,
                                       torch::Tensor bottom_mlp_output);
std::vector<torch::Tensor> dotBasedInteractBwdTorch(torch::Tensor input,
                                                    torch::Tensor upstreamGrad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dotBasedInteractFwd", &dotBasedInteractFwdTorch, "", py::arg("input"),
        py::arg("bottom_mlp_output"));
  m.def("dotBasedInteractBwd", &dotBasedInteractBwdTorch, "", py::arg("input"),
        py::arg("upstreamGrad"));
}
