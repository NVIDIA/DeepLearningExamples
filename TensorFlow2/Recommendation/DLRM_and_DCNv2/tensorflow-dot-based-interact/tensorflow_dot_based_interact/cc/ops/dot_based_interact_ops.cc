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


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("DotBasedInteract")
    .Attr("T: {float, half}")
    .Input("input: T")
    .Input("bottom_mlp_output: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      auto input = c->input(0);
      auto batch_size_dim = c->Dim(input, 0);
      int64 num_rows = c->Value(c->Dim(input, 1)); //number of categories
      int64 num_cols = c->Value(c->Dim(input, 2)); //embedding size
      int64 raw_output_size = ((num_rows * (num_rows - 1)) >> 1) + num_cols;
      int64 output_size = ((raw_output_size-1)/8 + 1)*8; //round up to multiple of 8
      auto output_size_dim = c->MakeDim(output_size);
      c->set_output(0, c->MakeShape({batch_size_dim, output_size_dim}));
      return Status();
    });

REGISTER_OP("DotBasedInteractGrad")
    .Attr("T: {float, half}")
    .Input("input: T")
    .Input("upstream_grad: T")
    .Output("grad: T")
    .Output("bottom_mlp_grad: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      auto input = c->input(0);
      auto batch_size_dim = c->Dim(input, 0);
      auto num_cols_dim = c->Dim(input, 2);
      c->set_output(0, input); //gradient w.r.t categoricals
      c->set_output(1, c->MakeShape({batch_size_dim, num_cols_dim})); //gradient w.r.t bottom mlp
      return Status();
    });
