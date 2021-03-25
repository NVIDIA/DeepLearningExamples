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


#ifndef KERNEL_DOT_BASED_INTERACT_AMPERE_H_
#define KERNEL_DOT_BASED_INTERACT_AMPERE_H_

void dotBasedInteractAmpereF16Fwd(const void *input,
                                         const void *bottom_mlp_output,
                                         void *output,
                                         uint batch_size,
                                         uint num_rows,
                                         uint num_cols,
                                         cudaStream_t stream);

void dotBasedInteractAmpereF16Bwd(const void *input,
                                         const void *upstream_grad,
                                         void *grad,
                                         void *bottom_mlp_grad,
                                         uint batch_size,
                                         uint num_rows,
                                         uint num_cols,
                                         cudaStream_t stream);

void dotBasedInteractAmpereTF32Fwd(const void *input,
                                          const void *bottom_mlp_output,
                                          void *output,
                                          uint batch_size,
                                          uint num_rows,
                                          uint num_cols,
                                          cudaStream_t stream);

void dotBasedInteractAmpereTF32Bwd(const void *input,
                                          const void *upstream_grad,
                                          void *grad,
                                          void *bottom_mlp_grad,
                                          uint batch_size,
                                          uint num_rows,
                                          uint num_cols,
                                          cudaStream_t stream);

#endif //KERNEL_DOT_BASED_INTERACT_AMPERE_H_
