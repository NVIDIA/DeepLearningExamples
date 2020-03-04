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
#pragma once

#include "fastertransformer/faster_transformer.h"

#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>
#include <iomanip>
#include <chrono>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;
using namespace fastertransformer;

template <typename T> class TransformerTrtTraits;

template <>
class TransformerTrtTraits<float>
{
  public:
    static const OperationType OpType = OperationType::FP32;
    static const nvinfer1::DataType DataType = nvinfer1::DataType::kFLOAT;
};

template <>
class TransformerTrtTraits<half>
{
  public:
    static const OperationType OpType = OperationType::FP16;
    static const nvinfer1::DataType DataType = nvinfer1::DataType::kFP16;
};

class Logger : public nvinfer1::ILogger
{
  public:
    Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
      if (severity > reportableSeverity) return;

      switch (severity)
      {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
      }
      std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};
static Logger gLogger(ILogger::Severity::kWARNING);

template <typename T>
class TransformerPlugin: public IPluginV2 
{
  public:
    TransformerPlugin(
        int hidden_dim, int head_num, int seq_len, int max_batch_size,
        const nvinfer1::Weights &w_attr_kernel_Q,
        const nvinfer1::Weights &w_attr_kernel_K,
        const nvinfer1::Weights &w_attr_kernel_V,
        const nvinfer1::Weights &w_attr_bias_Q,
        const nvinfer1::Weights &w_attr_bias_K,
        const nvinfer1::Weights &w_attr_bias_V,
        const nvinfer1::Weights &w_attr_output_kernel,
        const nvinfer1::Weights &w_attr_output_bias,
        const nvinfer1::Weights &w_attr_output_layernorm_beta,
        const nvinfer1::Weights &w_attr_output_layernorm_gamma,
        const nvinfer1::Weights &w_inter_kernel,
        const nvinfer1::Weights &w_inter_bias,
        const nvinfer1::Weights &w_output_kernel,
        const nvinfer1::Weights &w_output_bias,
        const nvinfer1::Weights &w_output_layernorm_beta,
        const nvinfer1::Weights &w_output_layernorm_gamma
        ): hidden_dim_(hidden_dim), head_num_(head_num), seq_len_(seq_len), max_batch_size_(max_batch_size) 
    {
      cudaMallocAndCopy(d_attr_kernel_Q_, w_attr_kernel_Q, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_kernel_K_, w_attr_kernel_K, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_kernel_V_, w_attr_kernel_V, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_bias_Q_, w_attr_bias_Q, hidden_dim);
      cudaMallocAndCopy(d_attr_bias_K_, w_attr_bias_K, hidden_dim);
      cudaMallocAndCopy(d_attr_bias_V_, w_attr_bias_V, hidden_dim);
      cudaMallocAndCopy(d_attr_output_kernel_, w_attr_output_kernel, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_output_bias_, w_attr_output_bias, hidden_dim);
      cudaMallocAndCopy(d_attr_output_layernorm_beta_, w_attr_output_layernorm_beta, hidden_dim);
      cudaMallocAndCopy(d_attr_output_layernorm_gamma_, w_attr_output_layernorm_gamma, hidden_dim);
      cudaMallocAndCopy(d_inter_kernel_, w_inter_kernel, hidden_dim * hidden_dim * 4);
      cudaMallocAndCopy(d_inter_bias_, w_inter_bias, hidden_dim * 4);
      cudaMallocAndCopy(d_output_kernel_, w_output_kernel, hidden_dim * hidden_dim * 4);
      cudaMallocAndCopy(d_output_bias_, w_output_bias, hidden_dim);
      cudaMallocAndCopy(d_output_layernorm_beta_, w_output_layernorm_beta, hidden_dim);
      cudaMallocAndCopy(d_output_layernorm_gamma_, w_output_layernorm_gamma, hidden_dim);
      /* should modify 0 to current device id */
      try
      {
        check_cuda_error(cublasCreate(&cublas_handle_));
        int device_id;
        check_cuda_error(cudaGetDevice(&device_id));
        allocator_ = new fastertransformer::Allocator<AllocatorType::CUDA>(device_id);
        encoder_transformer_ = new 
          BertEncoderTransformer<EncoderTraits_>(*allocator_, max_batch_size, seq_len, seq_len, head_num, hidden_dim / head_num);

        EncoderInitParam<T> encoder_param; //init param here
   
        encoder_param.attr_kernel_Q = d_attr_kernel_Q_;
        encoder_param.attr_kernel_K = d_attr_kernel_K_;
        encoder_param.attr_kernel_V = d_attr_kernel_V_;
        encoder_param.attr_bias_Q = d_attr_bias_Q_;
        encoder_param.attr_bias_K = d_attr_bias_K_;
        encoder_param.attr_bias_V = d_attr_bias_V_;
        encoder_param.attr_output_kernel = d_attr_output_kernel_;
        encoder_param.attr_output_bias = d_attr_output_bias_;
        encoder_param.attr_output_layernorm_beta = d_attr_output_layernorm_beta_;
        encoder_param.attr_output_layernorm_gamma = d_attr_output_layernorm_gamma_;
        encoder_param.inter_kernel = d_inter_kernel_;
        encoder_param.inter_bias = d_inter_bias_;
        encoder_param.output_kernel = d_output_kernel_;
        encoder_param.output_bias = d_output_bias_;
        encoder_param.output_layernorm_beta = d_output_layernorm_beta_;
        encoder_param.output_layernorm_gamma = d_output_layernorm_gamma_;
        encoder_param.cublas_handle = cublas_handle_;

        encoder_transformer_->initialize(encoder_param);
      }
      catch(std::runtime_error& error)
      {
        std::cout << error.what() << std::endl;
      }
    }
    TransformerPlugin(
        int hidden_dim, int head_num, int seq_len, int max_batch_size,
        const T* dp_attr_kernel_Q,
        const T* dp_attr_kernel_K,
        const T* dp_attr_kernel_V,
        const T* dp_attr_bias_Q,
        const T* dp_attr_bias_K,
        const T* dp_attr_bias_V,
        const T* dp_attr_output_kernel,
        const T* dp_attr_output_bias,
        const T* dp_attr_output_layernorm_beta,
        const T* dp_attr_output_layernorm_gamma,
        const T* dp_inter_kernel,
        const T* dp_inter_bias,
        const T* dp_output_kernel,
        const T* dp_output_bias,
        const T* dp_output_layernorm_beta,
        const T* dp_output_layernorm_gamma
        ): hidden_dim_(hidden_dim), head_num_(head_num), seq_len_(seq_len), max_batch_size_(max_batch_size)
    {
      cudaMallocAndCopy(d_attr_kernel_Q_, dp_attr_kernel_Q, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_kernel_K_, dp_attr_kernel_K, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_kernel_V_, dp_attr_kernel_V, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_bias_Q_, dp_attr_bias_Q, hidden_dim);
      cudaMallocAndCopy(d_attr_bias_K_, dp_attr_bias_K, hidden_dim);
      cudaMallocAndCopy(d_attr_bias_V_, dp_attr_bias_V, hidden_dim);
      cudaMallocAndCopy(d_attr_output_kernel_, dp_attr_output_kernel, hidden_dim * hidden_dim);
      cudaMallocAndCopy(d_attr_output_bias_, dp_attr_output_bias, hidden_dim);
      cudaMallocAndCopy(d_attr_output_layernorm_beta_, dp_attr_output_layernorm_beta, hidden_dim);
      cudaMallocAndCopy(d_attr_output_layernorm_gamma_, dp_attr_output_layernorm_gamma, hidden_dim);
      cudaMallocAndCopy(d_inter_kernel_, dp_inter_kernel, hidden_dim * hidden_dim * 4);
      cudaMallocAndCopy(d_inter_bias_, dp_inter_bias, hidden_dim * 4);
      cudaMallocAndCopy(d_output_kernel_, dp_output_kernel, hidden_dim * hidden_dim * 4);
      cudaMallocAndCopy(d_output_bias_, dp_output_bias, hidden_dim);
      cudaMallocAndCopy(d_output_layernorm_beta_, dp_output_layernorm_beta, hidden_dim);
      cudaMallocAndCopy(d_output_layernorm_gamma_, dp_output_layernorm_gamma, hidden_dim);

      try
      {
        check_cuda_error(cublasCreate(&cublas_handle_));
        /* should modify 0 to current device id */
        int device_id;
        check_cuda_error(cudaGetDevice(&device_id));
        allocator_ = new fastertransformer::Allocator<AllocatorType::CUDA>(device_id);
        encoder_transformer_ = new 
          BertEncoderTransformer<EncoderTraits_>(*allocator_, max_batch_size, seq_len, seq_len, head_num, hidden_dim / head_num);

        EncoderInitParam<T> encoder_param; //init param here
   
        encoder_param.attr_kernel_Q = d_attr_kernel_Q_;
        encoder_param.attr_kernel_K = d_attr_kernel_K_;
        encoder_param.attr_kernel_V = d_attr_kernel_V_;
        encoder_param.attr_bias_Q = d_attr_bias_Q_;
        encoder_param.attr_bias_K = d_attr_bias_K_;
        encoder_param.attr_bias_V = d_attr_bias_V_;
        encoder_param.attr_output_kernel = d_attr_output_kernel_;
        encoder_param.attr_output_bias = d_attr_output_bias_;
        encoder_param.attr_output_layernorm_beta = d_attr_output_layernorm_beta_;
        encoder_param.attr_output_layernorm_gamma = d_attr_output_layernorm_gamma_;
        encoder_param.inter_kernel = d_inter_kernel_;
        encoder_param.inter_bias = d_inter_bias_;
        encoder_param.output_kernel = d_output_kernel_;
        encoder_param.output_bias = d_output_bias_;
        encoder_param.output_layernorm_beta = d_output_layernorm_beta_;
        encoder_param.output_layernorm_gamma = d_output_layernorm_gamma_;
        encoder_param.cublas_handle = cublas_handle_;

        encoder_transformer_->initialize(encoder_param);
      }
      catch(std::runtime_error& error)
      {
        std::cout << error.what() << std::endl;
      }
    }

    ~TransformerPlugin() 
    {
      try{
        check_cuda_error(cudaFree(d_attr_kernel_Q_));
        check_cuda_error(cudaFree(d_attr_kernel_K_));
        check_cuda_error(cudaFree(d_attr_kernel_V_));
        check_cuda_error(cudaFree(d_attr_bias_Q_));
        check_cuda_error(cudaFree(d_attr_bias_K_));
        check_cuda_error(cudaFree(d_attr_bias_V_));
        check_cuda_error(cudaFree(d_attr_output_kernel_));
        check_cuda_error(cudaFree(d_attr_output_bias_));
        check_cuda_error(cudaFree(d_attr_output_layernorm_beta_));
        check_cuda_error(cudaFree(d_attr_output_layernorm_gamma_));
        check_cuda_error(cudaFree(d_inter_kernel_));
        check_cuda_error(cudaFree(d_inter_bias_));
        check_cuda_error(cudaFree(d_output_kernel_));
        check_cuda_error(cudaFree(d_output_bias_));
        check_cuda_error(cudaFree(d_output_layernorm_beta_));
        check_cuda_error(cudaFree(d_output_layernorm_gamma_));
        check_cuda_error(cublasDestroy(cublas_handle_));
        delete encoder_transformer_;
      }
      catch(std::runtime_error& error)
      {
        std::cout << error.what() << std::endl;
      }
    }

    virtual size_t getSerializationSize() const override {return 0;}
    virtual void serialize(void* buffer) const override {}

    int getNbOutputs() const override {return 1;}

    Dims getOutputDimensions(int index, const Dims* pInputDim, int nInputDim) override 
    {
      assert(index == 0 && nInputDim == 2);
      return DimsHW(seq_len_, hidden_dim_);
    }

    bool supportsFormat(nvinfer1::DataType type, PluginFormat format) const override 
    {
      return type == nvinfer1::DataType::kFLOAT && format == PluginFormat::kNCHW;
    }

    void configureWithFormat(const Dims* pInputDim, int nInputDim, const Dims* pOutputDim, 
        int nOutputDim, nvinfer1::DataType dataType, nvinfer1::PluginFormat pluginFormat, int maxBatchSize) override 
    {
      assert(dataType == nvinfer1::DataType::kFLOAT && pluginFormat == nvinfer1::PluginFormat::kNCHW);
      assert(nInputDim == 2);
      assert(pInputDim[0].nbDims == 2 && pInputDim[0].d[0] == seq_len_ && pInputDim[0].d[1] == hidden_dim_);
      assert(pInputDim[1].nbDims == 2 && pInputDim[1].d[0] == seq_len_ && pInputDim[1].d[1] == seq_len_);
      assert(nOutputDim == 1);
      assert(pOutputDim[0].nbDims == 2 && pOutputDim[0].d[0] == seq_len_ && pOutputDim[0].d[1] == hidden_dim_);
    }

    virtual int enqueue(int batch_size, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) override 
    {
      T* from_tensor = (T*) (inputs[0]);
      T* to_tensor = (T*) (inputs[0]);
      T* attr_mask = (T*) (inputs[1]);
      T* transformer_out = (T*) (outputs[0]);
      try
      {
        check_cuda_error(cublasSetStream(cublas_handle_, stream));
        encoder_transformer_->trt_initialize(from_tensor, to_tensor, attr_mask, transformer_out, stream, cublas_handle_);
        encoder_transformer_->forward();
      }
      catch(std::runtime_error& error)
      {
        std::cout << error.what() << std::endl;
      }
      return 0;
    }
    virtual size_t getWorkspaceSize(int nBatch) const override {return 0;}

    const char* getPluginType() const override {return "TransformerPlugin";}
    const char* getPluginVersion() const override {return "0";}

    IPluginV2* clone() const override 
    {
      return new TransformerPlugin(
          hidden_dim_, head_num_, seq_len_, max_batch_size_,  
          d_attr_kernel_Q_, 
          d_attr_kernel_K_, 
          d_attr_kernel_V_, 
          d_attr_bias_Q_, 
          d_attr_bias_K_, 
          d_attr_bias_V_,
          d_attr_output_kernel_, 
          d_attr_output_bias_, 
          d_attr_output_layernorm_beta_, 
          d_attr_output_layernorm_gamma_,
          d_inter_kernel_, 
          d_inter_bias_, 
          d_output_kernel_, 
          d_output_bias_,
          d_output_layernorm_beta_, 
          d_output_layernorm_gamma_
          );
    }
    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}

    static void cudaMallocAndCopy(T *&dpWeight, const nvinfer1::Weights &w, int nValue) 
    {
      assert(w.count == nValue);
      check_cuda_error(cudaMalloc(&dpWeight, nValue * sizeof(T)));
      check_cuda_error(cudaMemcpy(dpWeight, w.values, nValue * sizeof(T), cudaMemcpyHostToDevice));

      T* data = (T*)malloc(sizeof(T) * nValue);
      cudaMemcpy(data, dpWeight, sizeof(T) * nValue, cudaMemcpyDeviceToHost);

    }
    static void cudaMallocAndCopy(T*&dpWeight, const T *&dpWeightOld, int nValue) 
    {
      check_cuda_error(cudaMalloc(&dpWeight, nValue * sizeof(T)));
      check_cuda_error(cudaMemcpy(dpWeight, dpWeightOld, nValue * sizeof(T), cudaMemcpyDeviceToDevice));
    }

  private:
    int hidden_dim_ = 0, head_num_ = 0, seq_len_ = 0, max_batch_size_;
    T *d_attr_kernel_Q_ = NULL, *d_attr_kernel_K_ = NULL, *d_attr_kernel_V_ = NULL;
    T *d_attr_bias_Q_ = NULL, *d_attr_bias_K_ = NULL, *d_attr_bias_V_ = NULL;
    T *d_attr_output_kernel_ = NULL, *d_attr_output_bias_ = NULL;
    T *d_attr_output_layernorm_beta_ = NULL;
    T *d_attr_output_layernorm_gamma_ = NULL;
    T *d_inter_kernel_ = NULL, *d_inter_bias_ = NULL;
    T *d_output_kernel_ = NULL, *d_output_bias_ = NULL, *d_output_layernorm_beta_ = NULL, *d_output_layernorm_gamma_ = NULL;
    cublasHandle_t cublas_handle_;
    typedef BertEncoderTransformerTraits< TransformerTrtTraits<T>::OpType , cuda::OpenMultiHeadAttention> EncoderTraits_;
    BertEncoderTransformer<EncoderTraits_> *encoder_transformer_;
    fastertransformer::Allocator<AllocatorType::CUDA> *allocator_;
};
