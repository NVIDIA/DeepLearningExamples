# Mask R-CNN For PyTorch
This repository provides a script and recipe to train and infer on MaskRCNN to achieve state of the art accuracy, and is tested and maintained by NVIDIA.
 
## Table Of Contents
* [Model overview](#model-overview)
  * [Model Architecture](#model-architecture)  
  * [Default configuration](#default-configuration)
  * [Feature support matrix](#feature-support-matrix)
    * [Features](#features)
  * [Mixed precision training](#mixed-precision-training)
    * [Enabling mixed precision](#enabling-mixed-precision)
    * [Enabling TF32](#enabling-tf32)
  * [Performance Optimizations](#performance-optimizations)
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Parameters](#parameters)
  * [Command line options](#command-line-options)
  * [Getting the data](#getting-the-data)
    * [Dataset guidelines](#dataset-guidelines)
  * [Training process](#training-process)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA DGX A100 (8x A100 80GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-80gb)  
      * [Training accuracy: NVIDIA DGX-1 (8x V100 32GB)](#training-accuracy-nvidia-dgx-1-8x-v100-32gb)
      * [Training loss curves](#training-loss-curves)
      * [Training stability test](#training-stability-test)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA DGX A100 (8x A100 80GB)](#training-performance-nvidia-dgx-a100-8x-a100-80gb)
      * [Training performance: NVIDIA DGX-1 (8x V100 32GB)](#training-performance-nvidia-dgx-1-8x-v100-32gb)
    * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA DGX A100 (1x A100 80GB)](#inference-performance-nvidia-dgx-a100-1x-a100-80gb)
      * [Inference performance: NVIDIA DGX-1 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-1x-v100-32gb)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)
 
## Model overview
 
Mask R-CNN is a convolution based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). NVIDIA’s Mask R-CNN is an optimized version of [Facebook’s implementation](https://github.com/facebookresearch/maskrcnn-benchmark).This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 1.3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
 
The repository also contains scripts to interactively launch training, benchmarking and inference routines in a Docker container.
 
The major differences between the official implementation of the paper and our version of Mask R-CNN are as follows:
  - Mixed precision support with [PyTorch AMP](https://pytorch.org/docs/stable/amp.html).
  - Gradient accumulation to simulate larger batches.
  - Custom fused CUDA kernels for faster computations.
 
These techniques/optimizations improve model performance and reduce training time by a factor of 1.3x, allowing you to perform more efficient instance segmentation with no additional effort.
 
Other publicly available implementations of Mask R-CNN include:
  -  [NVIDIA TensorFlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/MaskRCNN)
  -  [NVIDIA TensorFlow2](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN)
  - [Matterport](https://github.com/matterport/Mask_RCNN)
  - [Tensorpack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)
  - [Google’s tensorflow model](https://github.com/tensorflow/models/tree/master/research/object_detection)
 
### Model architecture
 
Mask R-CNN builds on top of FasterRCNN adding an additional mask head for the task of image segmentation.
 
The architecture consists of following:
- R-50 backbone with FPN
- RPN head
- RoI ALign
- Bounding and classification box head
- Mask head
 
### Default Configuration
The default configuration of this model can be found at `pytorch/maskrcnn_benchmark/config/defaults.py`. The default hyper-parameters are as follows:
  - General:
    - Base Learning Rate set to 0.001
    - Global batch size set to 16 images
    - Steps set to 30000
    - Images re-sized with aspect ratio maintained and smaller side length between [800,1333]
    - Global train batch size - 16
    - Global test batch size - 8
 
  - Feature extractor:
    - Backend network set to Resnet50_conv4
    - First two blocks of backbone network weights are frozen
 
  - Region Proposal Network (RPN):
    - Anchor stride set to 16
    - Anchor sizes set to (32, 64, 128, 256, 512)
    - Foreground IOU Threshold set to 0.7, Background IOU Threshold set to 0.5
    - RPN target fraction of positive proposals set to 0.5
    - Train Pre-NMS Top proposals set to 12000
    - Train Post-NMS Top proposals set to 2000
    - Test Pre-NMS Top proposals set to 6000
    - Test Post-NMS Top proposals set to 1000
    - RPN NMS Threshold set to 0.7
 
  - RoI heads:
    - Foreground threshold set to 0.5
    - Batch size per image set to 512
    - Positive fraction of batch set to 0.25
 
This repository implements multi-gpu and gradient accumulation to support larger batches and mixed precision support. This implementation also includes the following optimizations.
  - Target generation - Optimized GPU implementation for generating binary mask ground truths from the list of polygon coordinates that exist in the dataset.
  - Custom CUDA kernels for:
    - Box Intersection over Union (IoU) computation
    - Proposal matcher
    - Generate anchor boxes
    - Pre NMS box selection - Selection of RoIs based on objectness score before NMS is applied.
 
    The source files can be found under `maskrcnn_benchmark/csrc/cuda`.
 
### Feature support matrix
 
The following features are supported by this model.  
 
| **Feature** | **Mask R-CNN** |
|:---------:|:----------:|
|Native AMP|Yes|
|Native DDP|Yes|
|Native NHWC|Yes|
 
#### Features
 
[AMP](https://pytorch.org/docs/stable/amp.html) is an abbreviation used for automatic mixed precision training.
 
[Native DDP](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html); [Apex DDP](https://nvidia.github.io/apex/parallel.html) where DDP stands for DistributedDataParallel and is used for multi-GPU training.
 
[NHWC](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html) is the channels last memory format for tensors.
  
### Mixed precision training
 
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
 
1.  Porting the model to use the FP16 data type where appropriate.
    
2.  Adding loss scaling to preserve small gradient values.
    
 
  
 
For information about:
 
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
    
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
  
 
#### Enabling mixed precision

In this repository, mixed precision training is enabled by the [PyTorch native AMP](https://pytorch.org/docs/stable/amp.html) library. PyTorch has an automatic mixed precision module that allows mixed precision to be enabled with minimal code changes.

Automatic mixed precision can be enabled with the following code changes:
```
# Create gradient scaler
scaler = torch.cuda.amp.GradScaler(init_scale=8192.0)

# Wrap the forward pass in torch.cuda.amp.autocast
with torch.cuda.amp.autocast():
  loss_dict = model(images, targets)

# Gradient scaling
scaler.scale(losses).backward()
scaler.step(optimizer)
scaler.update()
```
AMP can be enabled by setting `DTYPE` to `float16`.
 
#### Enabling TF32
 
 
TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 
 
TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.
 
For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.
 
TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.


### Performance Optimizations

[MLPerf Training](https://mlcommons.org/en/training-normal-11/) is an [ML Commons](https://mlcommons.org/en/) benchmark that measures how fast systems can train models to a target quality metric. MaskRCNN is one of the MLPerf training benchmarks which is improved every year. Some of the performance optimizations used in MLPerf can be introduced to this repository easily to gain significant training speedup. [Here](https://github.com/mlcommons/training_results_v1.1/tree/main/NVIDIA/benchmarks/maskrcnn/implementations/pytorch) is NVIDIA's MLPerf v1.1 submission codebase.

Listed below are some of the performance optimization tricks applied to this repository:

- Prefetcher: [PyTorch CUDA Streams](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html) are used to fetch the data required for the next iteration during the current iteration to reduce dataloading time before each iteration.
- pin_memory: Setting pin_memory can speed up host to device transfer of samples in dataloader. More details can be found in [this blog](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/).
- Hybrid Dataloader: Some dataloading is done on the CPU and the rest is on the GPU.
- FusedSGD: Replace SGD with Apex FusedSGD for training speedup.
- Native DDP: Use PyTorch [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
- Native NHWC: Switching from channels first (NCHW) memory format to NHWC (channels last) gives [better performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).

Increasing the local batch size and applying the above tricks gives ~2x speedup for end-to-end training time on 8 DGX A100s when compared to the old implementation.

## Setup
The following sections list the requirements in order to start training the Mask R-CNN model.
 
### Requirements
 
This repository contains `Dockerfile` which extends the PyTorch NGC container and encapsulates some dependencies.  Aside from these dependencies, ensure you have the following components:
  - [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
  - [PyTorch 21.12-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
-   Supported GPUs:
- [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
- [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
- [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)
 
  For more information about how to get started with NGC containers, see the
  following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning
  Documentation:
  - [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
  - [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/dgx/user-guide/index.html#accessing_registry)
  - [Running PyTorch](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html#running)
 
For those unable to use the [framework name] NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).
 
 
## Quick Start Guide
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Mask R-CNN model on the COCO 2017 dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.
 
 
### 1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/Segmentation/MaskRCNN
```
 
### 2. Download and preprocess the dataset.
This repository provides scripts to download and extract the COCO 2017 dataset.  Data will be downloaded to the `current working` directory on the host and extracted to a user-defined directory
 
To download, verify, and extract the COCO dataset, use the following scripts:
  ```
  ./download_dataset.sh <data/dir>
  ```
By default, the data is organized into the following structure:
  ```
  <data/dir>
    annotations/
      instances_train2017.json
      instances_val2017.json
    train2017/
      *.jpg
    val2017/
      *.jpg
  ```
 
### 3. Build the Mask R-CNN PyTorch NGC container.
```
cd pytorch/
bash scripts/docker/build.sh
```
 
### 4. Start an interactive session in the NGC container to run training/inference.
After you build the container image, you can start an interactive CLI session with  
```
bash scripts/docker/interactive.sh <path/to/dataset/>
```
The `interactive.sh` script requires that the location on the dataset is specified.  For example, `/home/<USER>/Detectron_PyT/detectron/lib/datasets/data/coco`
 
 
### 5. Start training.
```
bash scripts/train.sh
```
The `train.sh` script trains a model and performs evaluation on the COCO 2014 dataset. By default, the training script:
  - Uses 8 GPUs.
  - Saves a checkpoint every 2500 iterations and at the end of training. All checkpoints, evaluation results and training logs are saved to the `/results` directory (in the container which can be mounted to a local directory).
  - Mixed precision training with Tensor Cores is invoked by either adding `--amp` to the command line or `DTYPE \"float16\"` to the end of the above command as shown in the train script. This will override the default `DTYPE` configuration which is tf32 for Ampere and float32 for Volta.
  - Channels last memory format can be set using the `NHWC` flag which is set to `True` by default. Disabling this flag will run training using `NCHW` or channels first memory format.
 
  The `scripts/train.sh` script runs the following Python command:
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file “configs/e2e_mask_rcnn_R_50_FPN_1x.yaml”  
  ```
 
### 6. Start validation/evaluation.
  ```
  bash scripts/eval.sh
  ```
Model evaluation on a checkpoint can be launched by running the  `pytorch/scripts/eval.sh` script. The script requires:
- the location of the checkpoint folder to be specified and present within/mounted to the container.
- a text file named last_checkpoint which contains the path to the latest checkpoint. This mechanism is required in order to resume training from the latest checkpoint.
- The file last_checkpoint is automatically created at the end of the training process.
 
By default, evaluation is performed on the test dataset once training is complete. To skip evaluation at the end of training, issue the `--skip-test` flag.
 
Additionally, to perform evaluation after every epoch and terminate training on reaching a minimum required mAP score, set
- `PER_EPOCH_EVAL = True`
- `MIN_BBOX_MAP = <required value>`
- `MIN_MASK_MAP = <required value>`
 
### 7. Start inference/predictions.
 
Model predictions can be obtained on a test dataset and a model checkpoint by running the  `scripts/inference.sh <config/file/path>` script. The script requires:
  - the location of the checkpoint folder and dataset to be specified and present within/mounted to the container.
  - a text file named last_checkpoint which contains the path to the checkpoint.
 
For example:
```
bash scripts/inference.sh configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
```
 
Model prediction files get saved in the `<OUTPUT_DIR>/inference` directory and correspond to:

```
bbox.json - JSON file containing bounding box predictions
segm.json - JSON file containing mask predictions
predictions.pth - All prediction tensors computed by the model in torch.save() format
coco_results.pth - COCO evaluation results in torch.save() format - if --skip-eval is not used in the script above
```
 
To perform inference and skip computation of mAP scores, issue the `--skip-eval` flag. Performance is reported in seconds per iteration per GPU. The benchmarking scripts can be used to extract frames per second on training and inference.
 
## Advanced
The following sections provide greater details of the dataset, running training and inference, and the training results.
 
### Scripts and sample code
 
 
Descriptions of the key scripts and folders are provided below.
 
  
 
-   maskrcnn_benchmark - Contains scripts to build individual components of the model such as backbone, FPN, RPN, mask and bbox heads etc.,
-   download_dataset.sh - Launches download and processing of required datasets.
    
-   scripts/ - Contains shell scripts to launch data download, train the model and perform inferences.
    
 
  -   train.sh - Launches model training
      
  -   eval.sh  - Performs inference and computes mAP of predictions.
      
  -   inference.sh  - Performs inference on given data.
      
  -   train_benchmark.sh  - To benchmark training performance.
      
  -   inference_benchmark.sh  - To benchmark inference performance.
  -   docker/ - Scripts to build the docker image and to start an interactive session.   
    
-   tools/
    - train_net.py - End to end to script to load data, build and train the model.
    - test_net.py - End to end script to load data, checkpoint and perform inference and compute mAP score.
 
 
### Parameters
#### train_net.py script parameters
You can modify the training behaviour through the various flags in both the `train_net.py` script and through overriding specific parameters in the YAML config files. Flags in the `train_net.py` script are as follows:
  
  `--config_file` - path to config file containing model params
  
  `--skip-test` - skips model testing after training
  
  `--opts` - allows for you to override specific params in config file
 
For example:
```
python -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py \
    --config-file configs/e2e_faster_rcnn_R_50_FPN_1x.yaml \
    DTYPE "float16" \
    NHWC True \
    OUTPUT_DIR RESULTS \
    SOLVER.BASE_LR 0.002 \
    SOLVER.STEPS ‘(360000, 480000)’
```
  
### Command-line options
 
To see the full list of available options and their descriptions, use the -h or --help command line option, for example: 
 
`python tools/train_net.py --help`
 
 
### Getting the data
The Mask R-CNN model was trained on the [COCO 2017](http://cocodataset.org/#download) dataset.  This dataset comes with a training and validation set.  
 
This repository contains the `./download_dataset.sh`,`./verify_dataset.sh`, and `./extract_dataset.sh` scripts which automatically download and preprocess the training and validation sets.
 
#### Dataset guidelines
 
In order to run on your own dataset, ensure your dataset is present/mounted to the Docker container with the following hierarchy:
```
my_dataset/
  images_train/
  images_val/
  instances_train.json
  instances_val.json
```
and add it to `DATASETS` dictionary in `maskrcnn_benchmark/config/paths_catalog.py`
 
```
DATASETS = {
        "my_dataset_train": {
            "img_dir": "data/images_train",
            "ann_file": "data/instances_train.json"
        },
        "my_dataset_val": {
            "img_dir": "data/images_val",
            "ann_file": "data/instances_val.json"
        },
      }
```
```
python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> tools/train_net.py \
        --config-file <CONFIG? \
        DATASETS.TRAIN "(\"my_dataset_train\")"\
        DATASETS.TEST "(\"my_dataset_val\")"\
        DTYPE "float16" \
        OUTPUT_DIR <RESULTS> \
        | tee <LOGFILE>
```
 
### Training Process
Training is performed using the `tools/train_net.py` script along with parameters defined in the config file. The default config files can be found in the `pytorch/configs/` directory.
 
The `e2e_mask_rcnn_R_50_FPN_1x.yaml` file was used to gather accuracy and performance metrics. This configuration sets the following parameters:
  - Backbone weights to ResNet-50
  - Feature extractor set to ResNet-50 with Feature Pyramid Networks (FPN)
  - RPN uses FPN
  - RoI Heads use FPN
  - Dataset - COCO 2017
  - Base Learning Rate - 0.12
  - Global train batch size - 96
  - Global test batch size - 8
  - RPN batch size - 256
  - ROI batch size - 512
  - Solver steps - (12000, 16000)
  - Max iterations - 16667
  - Warmup iterations - 800
  - Warmup factor = 0.0001
    - Initial learning rate = Base Learning Rate x Warmup factor
 
The default feature extractor can be changed by setting `CONV_BODY` parameter in `yaml` file to any of the following:
  - R-50-C4
  - R-50-C5
  - R-101-C4
  - R-101-C5
  - R-101-FPN
 
The default backbone can be changed to a flavor of Resnet-50 or ResNet-101 by setting `WEIGHT` parameter in `yaml` file to any of the following:
  - "catalog://ImageNetPretrained/MSRA/R-50-GN"
  - "catalog://ImageNetPretrained/MSRA/R-101"
  - "catalog://ImageNetPretrained/MSRA/R-101-GN"
 
This script outputs results to the current working directory by default. However, this can be changed by adding `OUTPUT_DIR <DIR_NAME>` to the end of the default command. Logs produced during training are also stored in the `OUTPUT_DIR` specified. The training log will contain information about:
  - Loss, time per iteration, learning rate and memory metrics
  - performance values such as time per step
  - test accuracy and test performance values after evaluation
 
The training logs are located in the `<OUTPUT_DIR>/log` directory. The summary after each training epoch is printed in the following format:
  ```
  INFO:maskrcnn_benchmark.trainer:eta: 4:42:15  iter: 20  loss: 1.8236 (2.7274)  loss_box_reg: 0.0249 (0.0620)  loss_classifier: 0.6086 (1.2918)  loss_mask: 0.6996 (0.8026)  loss_objectness: 0.5373 (0.4787)  loss_rpn_box_reg: 0.0870 (0.0924)  time: 0.2002 (0.3765)  data: 0.0099 (0.1242)  lr: 0.014347  max mem: 3508
  ```
  The mean and median training losses are reported every 20 steps.
 
Multi-gpu and multi-node training is enabled with the PyTorch distributed launch module. The following example runs training on 8 GPUs:
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file \"configs/e2e_mask_rcnn_R_50_FPN_1x.yaml\"
  ```
 
We have tested batch sizes upto 12 on a 32GB V100 and 80GB A100 with mixed precision. The repository also implements gradient accumulation functionality to simulate bigger batches as follows:
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file \"configs/e2e_mask_rcnn_R_50_FPN_1x.yaml\" SOLVER.ACCUMULATE_GRAD True SOLVER.ACCUMULATE_STEPS 4
  ```
 
By default, training is performed using FP32 on Volta and TF32 on Ampere, however training time can be reduced further using tensor cores and mixed precision. This can be done by either adding `--amp` to the command line or `DTYPE \"float16\"` to override the respective parameter in the config file.
 
__Note__: When training a global batch size >= 32, it is recommended to add required warmup by additionally setting the following parameters:
  - `SOLVER.WARMUP_ITERS 625`
  - `SOLVER.WARMUP_FACTOR 0.01`
 
When experimenting with different global batch sizes for training and inference, make sure `SOLVER.IMS_PER_BATCH` and `TEST.IMS_PER_BATCH` are divisible by the number of GPUs.  
 
#### Other training options
A sample single GPU config is provided under `configs/e2e_mask_rcnn_R_50_FPN_1x_1GPU.yaml`

To train with smaller global batch sizes (32 or 64) use `configs/e2e_mask_rcnn_R_50_FPN_1x_bs32.yaml` and `configs/e2e_mask_rcnn_R_50_FPN_1x_bs64.yaml` respectively.
 
For multi-gpu runs, `-m torch.distributed.launch --nproc_per_node num_gpus` is added prior to `tools/train_net.py`.  For example, for an 8 GPU run:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file “configs/e2e_mask_rcnn_R_50_FPN_1x.yaml”   
```
 
Training is terminated when either the required accuracies specified on the command line are reached or if the number of training iterations specified is reached.
 
To terminate training on reaching target accuracy on 8 GPUs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file “configs/e2e_mask_rcnn_R_50_FPN_1x.yaml” PER_EPOCH_EVAL True MIN_BBOX_MAP 0.377 MIN_MASK_MAP 0.342
```
 
__Note__: The score is always the Average Precision(AP) at
  - IoU = 0.50:0.95
  - Area = all - include small, medium and large
  - maxDets = 100
 
## Performance
 
### Benchmarking
Benchmarking can be performed for both training and inference. Both scripts run the Mask R-CNN model using the parameters defined in `configs/e2e_mask_rcnn_R_50_FPN_1x.yaml`. You can specify whether benchmarking is performed in FP16, TF32 or FP32 by specifying it as an argument to the benchmarking scripts.
 
#### Training performance benchmark
Training benchmarking can performed by running the script:
```
scripts/train_benchmark.sh <float16/tf32/float32> <number of gpus> <NHWC True/False> <Hybrid dataloader True/False>
```
 
#### Inference performance benchmark
Inference benchmarking can be performed by running the script:
```
scripts/inference_benchmark.sh <float16/tf32/float32> <batch_size>
```
 
### Results
The following sections provide details on how we achieved our performance and accuracy in training and inference.
#### Training Accuracy Results
 
##### Training accuracy: NVIDIA DGX A100 (8x A100 80GB)
 
Our results were obtained by running the `scripts/train.sh` training script in the 21.12-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs.
 
| GPUs    | Batch size / GPU    | BBOX mAP - TF32| MASK mAP - TF32  | BBOX mAP - FP16| MASK mAP - FP16  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (TF32 to mixed precision)
| --| --| --  | -- | -- | -- | -- | -- | --
| 8 | 12 | 0.3765 | 0.3408 | 0.3763 | 0.3417 | 2.15 | 1.85 | 1.16x
 
##### Training accuracy: NVIDIA DGX-1 (8x V100 32GB)
 
Our results were obtained by running the `scripts/train.sh`  training script in the PyTorch 21.12-py3 NGC container on NVIDIA DGX-1 with 8x V100 32GB GPUs.
 
| GPUs    | Batch size / GPU    | BBOX mAP - FP32| MASK mAP - FP32  | BBOX mAP - FP16| MASK mAP - FP16  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (FP32 to mixed precision)
| --| --| --  | -- | -- | -- | -- | -- | --
| 8 | 12 | 0.3768 | 0.3415 | 0.3755 | 0.3403 | 5.58 | 3.37 | 1.65x

Note: Currently V100 32GB + FP32 + NHWC + Hybrid dataloader causes a slowdown. So for all V100 32GB FP32 runs hybrid dataloader and NHWC are disabled `NHWC=False HYBRID=False DTYPE=float32 bash scripts/train.sh`
 
##### Training loss curves
 
![Loss Curve](./img/loss_curve.png)
 
Here, multihead loss is simply the summation of losses on the mask head and the bounding box head.
 
 
##### Training Stability Test
The following tables compare mAP scores across 5 different training runs with different seeds.  The runs showcase consistent convergence on all 5 seeds with very little deviation.
 
| **Config** | **Seed 1** | **Seed 2** | **Seed 3** |  **Seed 4** | **Seed 5** | **Mean** | **Standard Deviation** |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- |
|  8 GPUs, final AP BBox  | 0.3764 | 0.3766 | 0.3767 | 0.3752  | 0.3768 | 0.3763 | 0.0006 |
| 8 GPUs, final AP Segm | 0.3414 | 0.3411 | 0.341 | 0.3407  | 0.3415 | 0.3411 | 0.0003 |
 
#### Training Performance Results
 
##### Training performance: NVIDIA DGX A100 (8x A100 80GB)
 
Our results were obtained by running the `scripts/train_benchmark.sh` training script in the 21.12-py3 NGC container on NVIDIA DGX A100 (8x A100 80GB) GPUs. Performance numbers in images per second were averaged over 500 iterations.
 
| GPUs   | Batch size / GPU   | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)   | Weak scaling - TF32    | Weak scaling - mixed precision
 --- | --- | ----- | ----- | --- | --- | ----- |
| 1 | 12 | 23 | 24 | 1.04 | 1 | 1 |
| 4 | 12 | 104 | 106 | 1.02 | 4.52 | 4.42 |
| 8 | 12 | 193 | 209 | 1.08 | 8.39 | 8.71 |
 
##### Training performance: NVIDIA DGX-1 (8x V100 32GB)
 
Our results were obtained by running the `scripts/train_benchmark.sh` training script in the 21.12-py3 NGC container on NVIDIA DGX-1 with (8x V100 32GB) GPUs. Performance numbers in images per second were averaged over 500 iterations.
 
| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision
| --- | --- | ----- | ----- | --- | --- | -----
| 1 | 12 | 12 | 16 | 1.33 | 1 | 1 |
| 4 | 12 | 44 | 71 | 1.61 | 3.67 | 4.44 |
| 8 | 12 | 85 | 135 | 1.59 | 7.08 | 8.44 |
 
Note: Currently V100 32GB + FP32 + NHWC + Hybrid dataloader causes a slowdown. So for all V100 32GB FP32 runs hybrid dataloader and NHWC are disabled `bash scripts/train_benchmark.sh fp32 <number of gpus> False False`
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
#### Inference performance results
 
##### Inference performance: NVIDIA DGX A100 (1x A100 80GB)
 
Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the PyTorch 21.12-py3 NGC container on NVIDIA DGX A100 (1x A100 80GB) GPU.

FP16 Inference Latency

| Batch size   | Throughput Avg   | Latency Avg (ms)    | Latency 90% (ms)    | Latency 95% (ms)    | Latency 99% (ms)
| --- | ----- | ----- | ----- | ----- | ----- |
|  1  | 23 | 34.91 | 33.87 | 33.95 | 34.15 |
|  2  | 26 | 59.31 | 57.80 | 57.99 | 58.27 |
|  4  | 31 | 101.46 | 99.24 | 99.51 | 99.86 |
|  8  | 31 | 197.57 | 193.82 | 194.28 | 194.77 |

TF32 Inference Latency

| Batch size   | Throughput Avg   | Latency Avg (ms)    | Latency 90% (ms)    | Latency 95% (ms)    | Latency 99% (ms)
| --- | ----- | ----- | ----- | ----- | ----- |
|  1  | 25 | 31.66 | 31.03 | 31.13 | 31.26 |
|  2  | 28 | 56.91 | 55.88 | 56.05 | 56.02 |
|  4  | 29 | 104.11 | 102.29 | 102.53 | 102.74 |
|  8  | 30 | 201.13 | 197.43 | 197.84 | 198.19 |

##### Inference performance: NVIDIA DGX-1 (1x V100 32GB)
 
Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the PyTorch 21.12-py3 NGC container on NVIDIA DGX-1 with 1x V100 32GB GPUs. 

FP16 Inference Latency

| Batch size   | Throughput Avg   | Latency Avg (ms)    | Latency 90% (ms)    | Latency 95% (ms)    | Latency 99% (ms)
| --- | ----- | ----- | ----- | ----- | ----- |
|  1  | 19 | 44.72 | 43.62 | 43.77 | 44.03 |
|  2  | 21 | 82.80 | 81.37 | 81.67 | 82.06 |
|  4  | 22 | 155.25 | 153.15 | 153.63 | 154.10 |
|  8  | 22 | 307.60 | 304.08 | 304.82 | 305.48 |

FP32 Inference Latency

| Batch size   | Throughput Avg   | Latency Avg (ms)    | Latency 90% (ms)    | Latency 95% (ms)    | Latency 99% (ms)
| --- | ----- | ----- | ----- | ----- | ----- |
|  1  | 16 | 52.78 | 51.87 | 52.16 | 52.43 |
|  2  | 17 | 100.81 | 99.19 | 99.67 | 100.15 |
|  4  | 17 | 202.05 | 198.84 | 199.98 | 200.92 |
|  8  | 18 | 389.99 | 384.29 | 385.77 | 387.66 |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
## Release notes
 
### Changelog

May 2022
- Update container to 21.12
- Add native NHWC, native DDP
- Replace SGD with FusedSGD
- Use cuda streams to prefetch dataloader
- Hybrid dataloader
- Use new training recipe

October 2021
- Replace APEX AMP with PyTorch native AMP
- Use opencv-python version 4.4.0.42

July 2021
- Update container
- Use native AMP
- Update dataset to coco 2017

June 2020
- Updated accuracy and performance tables to include A100 results

September 2019
  - Updates for PyTorch 1.2
  - Jupyter notebooks added

July 2019
  - Update AMP to new API
  - Update README
  - Download support from torch hub
  - Update default test batch size to 1/gpu
 
March 2019
  - Initial release
 
 
### Known Issues
Currently V100 32GB + FP32 + NHWC + Hybrid dataloader causes a slowdown. So for all V100 32GB FP32 runs hybrid dataloader and NHWC should be disabled.

