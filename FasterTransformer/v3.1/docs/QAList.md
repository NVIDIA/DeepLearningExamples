# Questions and Answers #

### 1. Who are the target users of FasterTransformer? ###

The users who require efficient transformer inference and flexibility. FasterTransformer provides flexible APIs and highly optimized kernels. Compare to the fastest solution, TensorRT demo BERT, the performance of FasterTransformer encoder is only little slower in some cases. Besides, FasterTransformer also provides supporting of translation and GPT-2. 

### 2. Which models can be supported in FasterTransformer? ###

Basically, FasterTransformer provides highly optimized transformer block. The users requiring such efficient transformer implementation can get benefit from FasterTransformer. For example, BERT inference, encoder-decoder architecture with transformer block. Besides, FasterTransformer also provides supporting of translation and GPT-2. 

### 3. Which frameworks can be supported in FasterTransformer? ###

FasterTransformer provides C API and TensorFlow/PyTorch OP. Users can use FasterTransformer directly on these frameworks. For other frameworks, users are also able to wrap the C++ codes to integrate FasterTransformer.

### 4. How to run multiple inference with one model in TensorFlow? ###

The simplest method is using the CUDA [Multi-Process Service (MPS)]( https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf), which is supported since Volta GPUs. 

Another method is using multi-threading on the same TensorFlow graph and session. Users can load the model in python and call the FasterTransformer OP thread by thread with same model graph. Note that running multiple thread on the same FasterTransformer OP may lead to dead lock, especially when there are lots of threads. 

### 5. Which GPUs are supported in FasterTransformer? ###

We have verified the correctness and performance for GPUs with Compute Compatibility >= 7.0 such as V100 and T4. A100 also works, but still have some performance issue for small batch size.

### 6. Do the users only be able to use the docker image we recommend? ###

Not yet. It is a suggestion but not limitation. We recommend using these docker image to build the project for the first time to prevent environment problems. The users can also build the project in their environment directly. 

### 7. Is there any requirement of CPU configuration for FasterTransformer execution? ###

FasterTransformer’s approach is to offload the computational workloads to GPUs with the memory operations overlapped with them. So FasterTransformer performance is mainly decided by what kinds of GPUs and I/O devices are used. However, when the batch size and sequence length are both small, kernel launching is the bottleneck and hence worse CPU may lead to worse performance.

### 8. How to load model into FasterTransformer? ###

In C, users need to load the model by themselves and copy into GPU memory. 

In TensorFlow or PyTorch, users can load the checkpoint and put the weight tensor into FasterTransformer directly. Users can also load the model in other formats, like numpy, and put them into FasterTransformer directly like the weight tensor. 


