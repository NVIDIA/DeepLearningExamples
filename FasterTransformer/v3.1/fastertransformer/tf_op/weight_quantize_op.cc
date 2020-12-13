/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fastertransformer/faster_transformer.h"
#include "fastertransformer/tf_op/common_op.h"

int index_CUBLASLT_ORDER_COL4_4R2_8C(int col_id, int row_id, int m_32){
  int new_col = col_id >> 5;
  int new_row =   //CUBLASLT_ORDER_COL4_4R2_8C
                  ////row_id/8 is the number of tile of (8 rows 32 columns) -- column-major
                  ////row_id%2 is even row, otherwise odd row
                  ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
                  (
                  ((((row_id >> 3) << 3) + ((row_id&1) << 2) + ((col_id&31) >> 3)) << 5) +
                  ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
                  ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
                  (((((col_id&7) >= 4)?4:0) + ((row_id&7) >> 1)) << 2) +
                  ////col_id%4 is the id of 4 cols
                  (col_id&3)
                  )
                  ;
  return new_col*m_32 + new_row;
}

int index_CUBLASLT_ORDER_COL32_2R_4R4(int col_id, int row_id, int m_32){
  int new_col = col_id >> 5;
  int row_in_tile = row_id & 31;
  int col_in_tile = col_id & 31;
  int new_row =   //CUBLASLT_ORDER_COL32_2R_4R4
                  (
                  ((row_id >> 5) << 10) +
                  //(((row%8)/2*4+row/8)*2+row%2)*32+col
                  (((((((row_in_tile&7)>>1)<<2)+(row_in_tile>>3))<<1)+(row_in_tile&1))<<5)+col_in_tile
                  )
                  ;
  return new_col*m_32 + new_row;
}

//be consistent with FasterTransformer
int8_t float_to_int8_rn_host(float x){
  int8_t res;
  int32_t tmp;
  if (x >= 0){
    tmp = int(x + 0.5);
    tmp = tmp > 127 ? 127 : tmp;
    res = int8_t(tmp);
  }
  else{
    tmp = int(x - 0.5);
    tmp = tmp < -127 ? -127 : tmp;
    res = int8_t(tmp);
  }
  return res;
}

template <typename T>
void quantization_CUBLASLT_ORDER_COL4_4R2_8C(T *dst, float* amaxs, const T* weight, const float* quant_max, const float *quant_min, int n, int k, bool per_channel_quantization){
  //quantization
  int8_t* int8_dst = (int8_t*)dst;
  float element;
  float amax;
  float amax_in_all = fabs(quant_max[0]);
  if (per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = fabs(quant_min[i]);
      if (fabs(quant_max[i]) > amaxs[i])
        amaxs[i] = fabs(quant_max[i]);
      if (amaxs[i] > amax_in_all)
        amax_in_all = amaxs[i];
    }
  } 
  if (!per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = amax_in_all;
    }
  }
  int idx_in_COL4;
  int tmp, tmpI;
  for (int col = 0 ; col < k ; col++){
    tmp = col*n;
    for (int row = 0 ; row < n ; row++){
      amax = amaxs[row];
      element = float(weight[tmp+row]);
      idx_in_COL4 = index_CUBLASLT_ORDER_COL4_4R2_8C(col, row, 32*n);
      int8_dst[idx_in_COL4] = float_to_int8_rn_host(element*127.0/amax);
    }
  }
}

template <typename T>
void quantization_CUBLASLT_ORDER_COL32_2R_4R4(T *dst, float* amaxs, const T* weight, const float* quant_max, const float *quant_min, int n, int k, bool per_channel_quantization){
  //quantization
  int8_t* int8_dst = (int8_t*)dst;
  float element;
  float amax;
  float amax_in_all = fabs(quant_max[0]);
  if (per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = fabs(quant_min[i]);
      if (fabs(quant_max[i]) > amaxs[i])
        amaxs[i] = fabs(quant_max[i]);
      if (amaxs[i] > amax_in_all)
        amax_in_all = amaxs[i];
    }
  }
  if (!per_channel_quantization){
    for (int i = 0 ; i < n ; i++){
      amaxs[i] = amax_in_all;
    }
  }
  int idx_in_COL32_2R_4R4;
  int tmp, tmpI;
  for (int col = 0 ; col < k ; col++){
    tmp = col*n;
    for (int row = 0 ; row < n ; row++){
      amax = amaxs[row];
      element = float(weight[tmp+row]);
      idx_in_COL32_2R_4R4 = index_CUBLASLT_ORDER_COL32_2R_4R4(col, row, 32*n);
      int8_dst[idx_in_COL32_2R_4R4] = float_to_int8_rn_host(element*127.0/amax);
    }
  }
}


namespace tensorflow
{
namespace
{
using CPUDevice = Eigen::ThreadPoolDevice;

REGISTER_OP("WeightQuantize")
    .Input("weight: T")
    .Input("quant_max: float")
    .Input("quant_min: float")
    .Output("output: T")
    .Output("output2: float")
    .Attr("T: {float, half}")
    .Attr("per_channel_quantization: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    });
template <typename Device, typename T>
class WeightQuantizeOp : public CommonOp<T>
{
public:
  explicit WeightQuantizeOp(OpKernelConstruction *context) : CommonOp<T>(context)
  {
     OP_REQUIRES_OK(context, context->GetAttr("per_channel_quantization", &per_channel_quantization_));
     use_ORDER_COL32_2R_4R4 = false;
#ifdef CUDA11_MODE
     int device{-1};
     cudaGetDevice(&device);
     cudaDeviceProp props;
     cudaGetDeviceProperties(&props, device);
     if (props.major * 10 + props.minor >= 80){
       use_ORDER_COL32_2R_4R4 = true;
     }
#endif 
  }

  void Compute(OpKernelContext *context) override
  {
    int rank = (int)context->input(0).dims();
    OP_REQUIRES(context, rank==2,
                errors::InvalidArgument("Invalid rank. The rank of weight should be 2 \
                                        ([n, k])"));

    k = (int)context->input(0).dim_size(0);
    n = (int)context->input(0).dim_size(1);

    OP_REQUIRES(context, context->num_inputs() == 3, errors::InvalidArgument("Less input arguments"));

    this->get_tensor(context, 0, &weight_);
    quant_max_ = reinterpret_cast<const float *>(context->input(1).flat<float>().data()); 
    OP_REQUIRES(context, quant_max_ != nullptr, errors::InvalidArgument("quant_max_ is null"));
    quant_min_ = reinterpret_cast<const float *>(context->input(2).flat<float>().data());
    OP_REQUIRES(context, quant_min_ != nullptr, errors::InvalidArgument("quant_min_ is null"));
    Tensor *output = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, {k, n}, &output));

    Tensor *output2 = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, {n}, &output2));
    transform_out = reinterpret_cast<T *>(output->flat<T>().data());
    transform_out2 = reinterpret_cast<float *>(output2->flat<float>().data());

    try
    {
      if (use_ORDER_COL32_2R_4R4)
        quantization_CUBLASLT_ORDER_COL32_2R_4R4(transform_out, transform_out2, weight_, quant_max_, quant_min_, n, k, per_channel_quantization_);
      else
        quantization_CUBLASLT_ORDER_COL4_4R2_8C(transform_out, transform_out2, weight_, quant_max_, quant_min_, n, k, per_channel_quantization_);
    }
    catch(std::runtime_error& error)
    {
      std::cout << errors::Internal(error.what());
      exit(-1);
    }
    catch(...)
    {
      std::cout << errors::Internal("Runtime error");
      exit(-1);
    }

  }

private:
  int n, k;
  const T *weight_;
  const float *quant_max_, *quant_min_;
  T* transform_out;
  float *transform_out2;
  bool use_ORDER_COL32_2R_4R4;
  bool per_channel_quantization_;
};

#define REGISTER_CPU(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("WeightQuantize").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      WeightQuantizeOp<CPUDevice, T>)
REGISTER_CPU(float);
REGISTER_CPU(Eigen::half);
#undef REGISTER_CPU

} //namespace
} //namespace tensorflow

