
# Jasper Inference For TensorRT

This is subfolder of the Jasper for PyTorch repository, tested and maintained by NVIDIA, and provides scripts to perform high-performance inference using NVIDIA TensorRT. Jasper is a neural acoustic model for speech recognition. Its network architecture is designed to facilitate fast GPU inference. More information about Jasper and its training and be found in the [Jasper PyTorch README](../README.md). 
NVIDIA TensorRT is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications.
After optimizing the compute-intensive acoustic model with NVIDIA TensorRT, inference throughput increased by up to 1.8x over native PyTorch. 



## Table Of Contents

- [Model overview](#model-overview)
   * [Model architecture](#model-architecture)
   * [TensorRT Inference pipeline](#tensorrt-inference-pipeline)
   * [Version Info](#version-info)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
   * [Scripts and sample code](#scripts-and-sample-code)
   * [Parameters](#parameters)
   * [TensorRT Inference Process](#tensorrt-inference-process)
   * [TensorRT Inference Benchmark Process](#tensorrt-inference-benchmark-process)
- [Performance](#performance)
   * [Results](#results)
      * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4)


## Model overview

### Model architecture
By default the model configuration is Jasper 10x5 with dense residuals. A Jasper BxR model has B blocks, each consisting of R repeating sub-blocks.
Each sub-block applies the following operations in sequence: 1D-Convolution, Batch Normalization, ReLU activation, and Dropout.
In the original paper Jasper is trained with masked convolutions, which masks out the padded part of an input sequence in a batch before the 1D-Convolution. 
For inference masking is not used. The reason for this is that in inference, the original mask operation does not achieve better accuracy than without the mask operation on the test and development dataset. However, no masking achieves better inference performance especially after TensorRT optimization.


### TensorRT Inference pipeline

The Jasper inference pipeline consists of 3 components: data preprocessor, acoustic model and greedy decoder. The acoustic model is the most compute intensive, taking more than 90% of the entire end-to-end pipeline. The acoustic model is the only component with learnable parameters and also what differentiates Jasper from the competition. So, we focus on the acoustic model for the most part.

For the non-TensorRT Jasper inference pipeline, all 3 components are implemented and run with native PyTorch. For the TensorRT inference pipeline, we show the speedup of running the acoustic model with TensorRT, while preprocessing and decoding are reused from the native PyTorch pipeline.

To run a model with TensorRT, we first construct the model in PyTorch, which is then exported into an ONNX file. Finally, a TensorRT engine is constructed from the ONNX file, serialized to TensorRT engine file, and also launched to do inference.

Note that TensorRT engine is being runtime optimized before serialization. TensorRT tries a vast set of options to find the strategy that performs best on user’s GPU - so it takes a few minutes. After the TensorRT engine file is created, it can be reused. 

### Version Info

The following software version configuration has been tested and known to work:

|Software|Version|
|--------|-------|
|Python|3.6.9|
|PyTorch|1.2.0|
|TensorRT|6.0.1.5|
|CUDA|10.1.243|

## Setup

The following section lists the requirements in order to start inference on the Jasper model with TensorRT.

### Requirements

This repository contains a `Dockerfile` which extends the PyTorch 19.10-py3 NGC container and encapsulates some dependencies. Ensure you have the following components:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.10-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
* [NVIDIA Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) based GPU
* [Pretrained Jasper Model Checkpoint](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16)

Required Python packages are listed in `requirements.txt` and `trt/requirements.txt`. These packages are automatically installed when the Docker container is built. To manually install them, run:


```bash
pip install -r requirements.txt
pip install -r trt/requirements.txt
```


## Quick Start Guide


Running the following scripts will build and launch the container containing all required dependencies for both TensorRT as well as native PyTorch. This is necessary for using inference with TensorRT and can also be used for data download, processing and training of the model.

1. Clone the repository.

      ```bash
      git clone https://github.com/NVIDIA/DeepLearningExamples
      cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
      ```
2. Build the Jasper PyTorch with TensorRT container:

      ```bash
      bash trt/scripts/docker/build.sh
      ```
3. Start an interactive session in the NGC docker container:

      ```bash
      bash trt/scripts/docker/launch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULT_DIR>
      ```

      Alternatively, to start a script in the docker container:

      ```bash
      bash trt/scripts/docker/aunch.sh <DATA_DIR> <CHECKPOINT_DIR> <RESULT_DIR> <SCRIPT_PATH>
      ```

      The `/datasets`, `/checkpoints`, `/results` directories will be mounted as volumes and mapped to the corresponding directories `<DATA_DIR>`, `<CHECKPOINT_DIR>`, `<RESULT_DIR>` on the host. **These three paths should be absolute and should already exist.** The contents of this repository will be mounted to the `/workspace/jasper` directory. Note that `<DATA_DIR>`, `<CHECKPOINT_DIR>`, and `<RESULT_DIR>` directly correspond to the same arguments in `scripts/docker/launch.sh` mentioned in the [Jasper PyTorch README](../README.md).

      Briefly, `<DATA_DIR>` should contain, or be prepared to contain a `LibriSpeech` sub-directory (created in [Acquiring Dataset](#acquiring-dataset)), `<CHECKPOINT_DIR>` should contain a PyTorch model checkpoint (`*.pt`) file obtained through training described in [Jasper PyTorch README](../README.md), and `<RESULT_DIR>` should be prepared to contain timing results, logs, serialized TensorRT engines, and ONNX files.

      4.  Acquiring dataset

      If LibriSpeech has already been downloaded and preprocessed as defined in the [Jasper PyTorch README](../README.md), no further steps in this subsection need to be taken.

      If LibriSpeech has not been downloaded already, note that only a subset of LibriSpeech is typically used for inference (`dev-*` and `test-*`). To acquire the inference subset of LibriSpeech run the following commands inside the container (does not require GPU):

      ```bash
      bash trt/scripts/download_inference_librispeech.sh
      ```

      Once the data download is complete, the following folders should exist:

      * `/datasets/LibriSpeech/`
         * `dev-clean/`
         * `dev-other/`
         * `test-clean/`
         * `test-other/`

      Next, preprocessing the data can be performed with the following command:

      ```bash
      bash trt/scripts/preprocess_inference_librispeech.sh
      ```

      Once the data is preprocessed, the following additional files should now exist:
      * `/datasets/LibriSpeech/`
         * `librispeech-dev-clean-wav.json`
         * `librispeech-dev-other-wav.json`
         * `librispeech-test-clean-wav.json`
         * `librispeech-test-other-wav.json`
         * `dev-clean-wav/`
         * `dev-other-wav/`
         * `test-clean-wav/`
         * `test-other-wav/`

5. Start TensorRT inference prediction

      Inside the container, use the following script to run inference with TensorRT.
      ```bash
      export CHECKPOINT=<CHECKPOINT>
      export TRT_PRECISION=<PRECISION>
      export PYTORCH_PRECISION=<PRECISION>
      export TRT_PREDICTION_PATH=<TRT_PREDICTION_PATH>
      bash trt/scripts/trt_inference.sh
      ```
      A pretrained model checkpoint can be downloaded from [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16). 
      More details can be found in [Advanced](#advanced) under [Scripts and sample code](#scripts-and-sample-code), [Parameters](#parameters) and [TensorRT Inference process](#trt-inference).

6.  Start TensorRT inference benchmark

      Inside the container, use the following script to run inference benchmark with TensorRT.
      ```bash
      export CHECKPOINT=<CHECKPOINT>
      export NUM_STEPS=<NUM_STEPS>
      export NUM_FRAMES=<NUM_FRAMES>
      export BATCH_SIZE=<BATCH_SIZE>
      export TRT_PRECISION=<PRECISION>
      export PYTORCH_PRECISION=<PRECISION>
      export CSV_PATH=<CSV_PATH>
      bash trt/scripts/trt_inference_benchmark.sh
      ```
      A pretrained model checkpoint can be downloaded from the [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16). 
      More details can be found in [Advanced](#advanced) under [Scripts and sample code](#scripts-and-sample-code), [Parameters](#parameters) and [TensorRT Inference Benchmark process](#trt-inference-benchmark).

7. Start Jupyter notebook to run inference interactively

      The Jupyter notebook  is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.
      The notebook which is located at `notebooks/JasperTRT.ipynb` offers an interactive method to run the Steps 2,3,4,5. In addition, the notebook shows examples how to use TensorRT to transcribe a single audio file into text. To launch the application please follow the instructions under [../notebooks/README.md](../notebooks/README.md). 
      A pretrained model checkpoint can be downloaded from [NGC model repository](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16). 


## Advanced
The following sections provide greater details on inference benchmarking with TensorRT and show inference results

### Scripts and sample code
In the `trt/` directory, the most important files are:
* `Dockerfile`: Container to run Jasper inference with TensorRT.
* `requirements.py`: Python package dependencies. Installed when building the Docker container.
* `perf.py`: Entry point for inference pipeline using TensorRT.
* `perfprocedures.py`: Contains functionality to run inference through both the PyTorch model and TensorRT Engine, taking runtime measurements of each component of the inference process for comparison.
* `trtutils.py`: Helper functions for TensorRT components of Jasper inference.
* `perfutils.py`: Helper functions for non-TensorRT components of Jasper inference.
* `onnx-trt.patch`: Used to enable Onnx and TensorRT with dynamic shapes.

The `trt/scripts/` directory has one-click scripts to run supported functionalities, such as:

* `download_librispeech.sh`: Downloads LibriSpeech inference dataset.
* `preprocess_librispeech.sh`: Preprocess LibriSpeech raw data files to be ready for inference.
* `trt_inference_benchmark.sh`: Benchmarks and compares TensorRT and PyTorch inference pipelines using the `perf.py` script.
* `trt_inference.sh`: Runs TensorRT and PyTorch inference using the `trt_inference_benchmark.sh` script.
* `walk_benchmark.sh`: Illustrates an example of using `trt/scripts/trt_inference_benchmark.sh`, which *walks* a variety of values for `BATCH_SIZE` and `NUM_FRAMES`.
* `docker/`: Contains the scripts for building and launching the container.


### Parameters

The list of parameters available for `trt/scripts/trt_inference_benchmark.sh` is:

```
Required:
--------
CHECKPOINT: Model checkpoint path

Arguments with Defaults:
--------
DATA_DIR: directory of the dataset (Default: `/datasets/Librispeech`)
DATASET: name of dataset to use (default: `dev-clean`)
RESULT_DIR: directory for results including TensorRT engines, ONNX files, logs, and CSVs (default: `/results`)
CREATE_LOGFILE: boolean that indicates whether to create log of session to be stored in `$RESULT_DIR` (default: "true")
CSV_PATH: file to store CSV results (default: `/results/res.csv`)
TRT_PREDICTION_PATH: file to store inference prediction results generated with TensorRT (default: `none`)
PYT_PREDICTION_PATH: file to store inference prediction results generated with native PyTorch (default: `none`)
VERBOSE: boolean that indicates whether to verbosely describe TensorRT engine building/deserialization and TensorRT inference (default: "false")
TRT_PRECISION: "fp32" or "fp16". Defines which precision kernels will be used for TensorRT engine (default: "fp32")
PYTORCH_PRECISION: "fp32" or "fp16". Defines which precision will be used for inference in PyTorch (default: "fp32")
NUM_STEPS: Number of inference steps. If -1 runs inference on entire dataset (default: 100)
BATCH_SIZE: data batch size (default: 64)
NUM_FRAMES: cuts/pads all pre-processed feature tensors to this length. 100 frames ~ 1 second of audio (default: 512)
FORCE_ENGINE_REBUILD: boolean that indicates whether an already-built TensorRT engine of equivalent precision, batch-size, and number of frames should not be used. Engines are specific to the GPU, library versions, TensorRT versions, and CUDA versions they were built in and cannot be used in a different environment. (default: "true")
USE_DYNAMIC_SHAPE: if 'yes' uses dynamic shapes (default: ‘yes’). Dynamic shape is always preferred since it allows to reuse engines.
```

The complete list of parameters available for `trt/scripts/trt_inference.sh` is the same as `trt/scripts/trt_inference_benchmark.sh` only with different default input arguments. In the following, only the parameters with different default values are listed:

```
TRT_PREDICTION_PATH: file to store inference prediction results generated with TensorRT (default: `/results/trt_predictions.txt`)
PYT_PREDICTION_PATH: file to store inference prediction results generated with native PyTorch (default: `/results/pyt_predictions.txtone`)
NUM_STEPS: Number of inference steps. If -1 runs inference on entire dataset (default: -1)
BATCH_SIZE: data batch size (default: 1)
NUM_FRAMES: cuts/pads all pre-processed feature tensors to this length. 100 frames ~ 1 second of audio (default: 3600)
```

### TensorRT Inference Benchmark process

The inference benchmarking is performed on a single GPU by ‘trt/scripts/trt_inference_benchmark.sh’ which delegates to `trt/perf.py`,  which takes the following steps:


1. Construct Jasper acoustic model in PyTorch.

2. Construct TensorRT Engine of Jasper acoustic model

   1. Perform ONNX export on the PyTorch model, if its ONNX file does not already exist.

	2. Construct TensorRT engine from ONNX export, if a saved engine file does not already exist or `FORCE_ENGINE_REBUILD` is `true`.

3. For each batch in the dataset, run inference through both the PyTorch model and TensorRT Engine, taking runtime measurements of each component of the inference process.

4. Compile performance and WER accuracy results in CSV format, written to `CSV_PATH` file.

`trt/perf.py` utilizes `trt/trtutils.py` and `trt/perfutils.py`, helper functions for TensorRT and non-TensorRT components of Jasper inference respectively.

### TensorRT Inference process

The inference is performed by `trt/scripts/trt_inference.sh` which delegates to `trt/scripts/trt_inference_benchmark.sh`. The script runs on a single GPU. To do inference prediction on the entire dataset `NUM_FRAMES` is set to 3600, which roughly corresponds to 36 seconds. This covers the longest sentences in both LibriSpeech dev and test dataset. By default, `BATCH_SET` is set to 1 to simulate the online inference scenario in deployment. Other batch sizes can be tried by setting a different value to this parameter. By default `TRT_PRECISION` is set to full precision and can be changed by setting `export TRT_PRECISION=fp16`. The prediction results are stored at `/results/trt_predictions.txt` and `/results/pyt_predictions.txt`.



## Performance

To benchmark the inference performance on a specific batch size and audio length refer to [Quick-Start-Guide](#quick-start-guide). To do a sweep over multiple batch sizes and audio durations run:
```bash
bash trt/scripts/walk_benchmark.sh
```
The results are obtained by running inference on LibriSpeech dev-clean dataset on a single T4 GPU using half precision with AMP. We compare the throughput of the acoustic model between TensorRT and native PyTorch.   

### Results



#### Inference performance: NVIDIA T4

| Sequence Length (in seconds) | Batch size | PyTorch FP16 Throughput (#sequences/second) Percentiles |     	|     	|     	| TensorRT FP16 Throughput (#sequences/second) Percentiles |     	|     	|     	| PyT/TRT Speedup |
|---------------|------------|---------------------|---------|---------|---------|-----------------|---------|---------|---------|-----------------|
|           	|        	| 90%             	| 95% 	| 99% 	| Avg 	| 90%         	| 95% 	| 99% 	| Avg 	|             	|
|2|1|71.002|70.897|70.535|71.987|42.974|42.932|42.861|43.166|1.668|
||2|136.369|135.915|135.232|139.266|81.398|77.826|57.408|81.254|1.714|
||4|231.528|228.875|220.085|239.686|130.055|117.779|104.529|135.660|1.767|
||8|310.224|308.870|289.132|316.536|215.401|202.902|148.240|228.805|1.383|
||16|389.086|366.839|358.419|401.267|288.353|278.708|230.790|307.070|1.307|
|7|1|61.792|61.273|59.842|63.537|34.098|33.963|33.785|34.639|1.834|
||2|93.869|92.480|91.528|97.082|59.397|59.221|51.050|60.934|1.593|
||4|113.108|112.950|112.531|114.507|66.947|66.479|59.926|67.704|1.691|
||8|118.878|118.542|117.619|120.367|83.208|82.998|82.698|84.187|1.430|
||16|122.909|122.718|121.547|124.190|102.212|102.000|101.187|103.049|1.205|
|16.7|1|38.665|38.404|37.946|39.363|21.267|21.197|21.127|21.456|1.835|
||2|44.960|44.867|44.382|45.583|30.218|30.156|29.970|30.679|1.486|
||4|47.754|47.667|47.541|48.287|29.146|29.079|28.941|29.470|1.639|
||8|51.051|50.969|50.620|51.489|37.565|37.497|37.373|37.834|1.361|
||16|53.316|53.288|53.188|53.773|45.217|45.090|44.946|45.560|1.180|
