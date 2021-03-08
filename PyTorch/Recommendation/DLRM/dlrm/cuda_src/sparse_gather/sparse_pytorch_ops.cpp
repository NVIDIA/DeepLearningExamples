#include <torch/extension.h>

torch::Tensor gather_gpu_fwd(torch::Tensor input, torch::Tensor weight);
void gather_gpu_bwd_fuse_sgd(const torch::Tensor grad, const torch::Tensor indices, float lr, torch::Tensor weight);
torch::Tensor gather_gpu_bwd(const torch::Tensor grad, const torch::Tensor indices, const int num_features);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_gpu_fwd", &gather_gpu_fwd, "Embedding gather", py::arg("indices"), py::arg("weight"));
  m.def("gather_gpu_bwd_fuse_sgd", &gather_gpu_bwd_fuse_sgd, "Embedding gather backward with fused plain SGD",
        py::arg("grad"), py::arg("indices"), py::arg("lr"), py::arg("weight"));
  m.def("gather_gpu_bwd", &gather_gpu_bwd, "Embedding gather backward",
        py::arg("grad"), py::arg("indices"), py::arg("num_features"));
}
