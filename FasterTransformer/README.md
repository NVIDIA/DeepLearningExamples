Faster Transformer
===================
## What is it?
The Faster Transformer implements an equivalent but highly optimized BERT transformer layer for inference. On Volta and Turing GPUs, FP16 precision is used automatically to access the computing power of tensor cores.

Faster Transformer is built on top of the CUDA and cuBLAS. It supports sequence lengths that are larger than 3 and smaller or equal to 1024. Two key parameters of the transformer layer, the number of heads and the size of each head, are passed in runtime. Thus, not only the BERT Base (12 heads *  64 per head) , but also customized models like 4 heads * 32 per head and 8 heads * 96 per heads, are well supported. Our implementation shows good speedups on both small and large batch size cases. 

C++ API, TensorRT plugin, and TensorFlow OP wrapper are available. You can easily integrate this optimized transformer layer into your TensorFlow or other inference service codes that built in native C++ or TensorRT. In addition to codes that illustrate the API invocations, we also provide a simple end-to-end BERT TensorFlow inference sample.

## Environment requirements
* CMake >= 3.8
* CUDA 10.0
* Python 2.7
* Tensorflow 1.13
* TensorRT 5.1.5
* The project is tested in nvidia/cuda 10.0-cudnn7-devel-ubuntu16.04 docker image. If you encountered compiling errors, try to compile with this docker image.

## Performance ##
* CPU: Intel(R) Xeon(R) Gold 6132 CPU @ 2.60GHz
* T4 (with mclk 5000MHz, pclk 1590MHz)  
* P4 (with mclk 2999MHz, pclk 1531MHz)  
* V100 (with mclk 877MHz, pclk 1380MHz)  

When batch size equals to 1, the Tensorflow execution time really depends on the CPU you are using. 

We only report the faster transformer performance here. 

The performance of the faster transformer mainly depends on GPU. The execution time is stable.


| <batch_size, layers, seq_len, head_num, size_per_head> | P4 FP32 (in ms) | T4 FP32 (in ms)| T4 FP16 (in ms)|
|:-------------:|:-------------:|:---------:|:-----------:|
| (1, 12, 32, 12, 64)  | 3.43  | 2.74 | 1.56 |
| (1, 12, 64, 12, 64)  | 4.04 | 3.64 | 1.77 | 
| (1, 12, 128, 12, 64) | 6.22 | 5.93 | 2.23 |


For large batch size case, we report both Tensorflow XLA and faster transformer's performance.

| <batch_size, layers, seq_len, head_num, size_per_head> | Tensorflow XLA on V100 FP16 (in ms)| Faster Transformer V100 FP16 (in ms) | Speedup |
|:-------------:|:-------------:|:---------:|:-----------:|
| (100, 12, 32, 12, 64)  | 13.96  | 9.57 | 1.459 |
| (200, 12, 32, 12, 64)  | 26.47  | 18.37 | 1.44 |
| (300, 12, 32, 12, 64)  | 38.4  | 27.41 | 1.401 |
| (400, 12, 32, 12, 64)  | 49.65  | 35.63 | 1.393 |
| (500, 12, 32, 12, 64)  | 62.2  | 44.57 | 1.396 |

| <batch_size, layers, seq_len, head_num, size_per_head> | Tensorflow XLA on V100 FP16 (in ms)| Faster Transformer V100 FP16 (in ms) | Speedup |
|:-------------:|:-------------:|:---------:|:-----------:|
| (100, 12, 32, 4, 32)  | 3.49  | 1.73 | 2.017 |
| (200, 12, 32, 4, 32)  | 4.9  | 2.55 | 1.922 |
| (300, 12, 32, 4, 32)  | 6.35  | 3.356 | 1.892 |
| (400, 12, 32, 4, 32)  | 8  | 4.31 | 1.856 |
| (500, 12, 32, 4, 32)  | 9.93  | 5.13 | 1.936 |

## Directory Structure
```
/fastertransformer: source code of transformer
   |--/cuda: some CUDA kernels and multi-head attention implementation, both are compiled with cuda/cuBLAS. 
   |--/tf_op: custom Tensorflow OP implementation
   |--/trt_plugin: TensorRT plugin implementation
/sample: c++ and tensorflow transformer interface samples
   |--/cpp: both FP16 and FP32 c++ interface samples
   |--/tensorflow_bert: samples that show of how to integrate our Tensorflow OP into the open source BERT model for sentence (and sentence-pair) classification tasks (GLUE), the samples support both FP16 and FP32, see readme file within this folder more details
   |--/tensorflow: both FP16 and FP32 tensorflow OP samples
   |--/tensorRT: both FP16 and FP32 tensorRT plugin samples
/tools/gemm_test: loop over all GEMM algorithms to pick the best one
```

## How to build?
### Init Git ###
```shell
$ git submodule init
$ git submodule update
```

### Build with Release ###
```shell
$ mkdir -p build
$ cd build
$ cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release .. # C++ only
$ cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON -DTRT_PATH=/myspace/TensorRT-5.1.5.0 .. # TensorRT mode
$ cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python2.7/dist-packages/tensorflow .. # Tensorflow mode
$ cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_TRT=ON -DTRT_PATH=/myspace/TensorRT-5.1.5.0 -DBUILD_TF=ON -DTF_PATH=/usr/local/lib/python2.7/dist-packages/tensorflow .. # C++, TensorRT and Tensorflow 
$ make
```

Note: xx is the compute capability of your GPU. For example, 60 (P40) or 61 (P4) or 70 (V100) or 75(T4).
### Execute demos ###
```shell
$ To achieve the best performance, please execute step1 and step2 together when you test a new model.
$ Step1 Generate the gemm_config.in file under the path build to pick GEMM algorithms for the best performance. 
$ ./build/bin/gemm_fp16(32) <batch_size> <seq_len> <head_num> <size_per_head>
$ Step2 Execute demos
$ 1. Tensorflow demos: python build/transformer_fp16(32).py <batch_size> <num_layers> <seq_len> <head_num> <size_per_head>
$ 2. c++ demos: ./build/bin/transformer_fp16(32) <batch_size> <num_layerse> <seq_len> <head_num> <size_per_head>
$ 3. TensorRT demos: ./build/bin/transformer_trt <batch_size> <num_layerse> <seq_len> <head_num> <size_per_head> fp16(fp32)
```

### Useful sample code ###
```shell
$ 1. sample/tensorflow/transformer_fp32.py: transformer_layer Tensorflow FP32 OP call, time measurement, timeline generation
$ 2. sample/tensorflow/transformer_fp16.py: transformer_layer Tensorflow FP16 OP call, time measurement, timeline generation
$ 3. sample/tensorflow/error_check.py: how to catch custom OP runtime errors
$ 4. sample/cpp/transformer_fp32.cc: transformer layer C++ FP32 sample
$ 5. sample/cpp/transformer_fp16.cc: transformer layer C++ FP16 sample
$ 6. sample/tensorRT/transformer_trt.cc: transformer layer tensorRT FP32/FP16 sample
$ 7. tools/gemm_test/gemm_fp16.cu: loop over all cublas FP16 GEMM algorithms and pick the best one
$ 8. tools/gemm_test/gemm_fp32.cu: loop over all cublas FP32 GEMM algorithms and pick the best one
```

