#include <torch/extension.h>

torch::Tensor gatherGPUFusedFwdTorch(torch::Tensor embedding,
                                       torch::Tensor indices,
                                       torch::Tensor offsets,
                                       bool amp_train);

torch::Tensor gatherGPUFusedBwdTorch(torch::Tensor embedding,
                                       torch::Tensor indices,
                                       torch::Tensor offsets,
                                       torch::Tensor upstreamGrad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_gpu_fused_fwd", &gatherGPUFusedFwdTorch, "", py::arg("embedding"),
                                                              py::arg("indices"),
                                                              py::arg("offsets"),
                                                              py::arg("amp_train"));
  m.def("gather_gpu_fused_bwd", &gatherGPUFusedBwdTorch, "", py::arg("embedding"),
                                                              py::arg("indices"),
                                                              py::arg("offsets"),
                                                              py::arg("upstreamGrad"));
}
