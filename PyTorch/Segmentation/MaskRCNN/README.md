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
* [Setup](#setup)
  * [Requirements](#requirements)
* [Quick start guide](#quick-start-guide)
* [Advanced](#advanced)
  * [Command line arguments](#command-line-arguments)
  * [Getting the data](#getting-the-data)
    * [Dataset guidelines](#dataset-guidelines)
  * [Training process](#training-process)
* [Performance](#performance)
  * [Benchmarking](#benchmarking)
    * [Training performance benchmark](#training-performance-benchmark)
    * [Inference performance benchmark](#inference-performance-benchmark)
  * [Results](#results)
    * [Training accuracy results](#training-accuracy-results)
      * [Training accuracy: NVIDIA DGX A100 (8x A100 40GB)](#training-accuracy-nvidia-dgx-a100-8x-a100-40gb)  
      * [Training accuracy: NVIDIA DGX-1 (8x V100 16GB)](#training-accuracy-nvidia-dgx-1-8x-v100-16gb)
      * [Training loss curves](#training-loss-curves)
      * [Training stability test](#training-stability-test)
    * [Training performance results](#training-performance-results)
      * [Training performance: NVIDIA DGX A100 (8x A100 40GB)](#training-performance-nvidia-dgx-a100-8x-a100-40gb)
      * [Training performance: NVIDIA DGX-1 (8x V100 16GB)](#training-performance-nvidia-dgx-1-8x-v100-16gb)
      * [Training performance: NVIDIA DGX-2 (16x V100 32GB)](#training-performance-nvidia-dgx-2-16x-v100-32gb)
    * [Inference performance results](#inference-performance-results)
      * [Inference performance: NVIDIA DGX A100 (1x A100 40GB)](#inference-performance-nvidia-dgx-a100-1x-a100-40gb)
      * [Inference performance: NVIDIA DGX-1 (1x V100 16GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
      * [Inference performance: NVIDIA DGX-2 (1x V100 32GB)](#inference-performance-nvidia-dgx-1-1x-v100-16gb)
* [Release notes](#release-notes)
  * [Changelog](#changelog)
  * [Known issues](#known-issues)
 
## Model overview
 
Mask R-CNN is a convolution based neural network for the task of object instance segmentation. The paper describing the model can be found [here](https://arxiv.org/abs/1703.06870). NVIDIA’s Mask R-CNN 19.2 is an optimized version of [Facebook’s implementation](https://github.com/facebookresearch/maskrcnn-benchmark).This model is trained with mixed precision using Tensor Cores on Volta, Turing, and the NVIDIA Ampere GPU architectures. Therefore, researchers can get results 1.3x faster than training without Tensor Cores, while experiencing the benefits of mixed precision training. This model is tested against each NGC monthly container release to ensure consistent accuracy and performance over time.
 
The repository also contains scripts to interactively launch training, benchmarking and inference routines in a Docker container.
 
The major differences between the official implementation of the paper and our version of Mask R-CNN are as follows:
  - Mixed precision support with [PyTorch AMP](https://github.com/NVIDIA/apex).
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
|APEX AMP|Yes|
|APEX DDP|Yes|
 
#### Features
 
[APEX](https://github.com/NVIDIA/apex) is a PyTorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training, whereas [AMP](https://nvidia.github.io/apex/amp.html) is an abbreviation used for automatic mixed precision training.
 
[DDP](https://nvidia.github.io/apex/parallel.html) stands for DistributedDataParallel and is used for multi-GPU training.
 
  
### Mixed precision training
 
Mixed precision is the combined use of different numerical precisions in a computational method. [Mixed precision](https://arxiv.org/abs/1710.03740) training offers significant computational speedup by performing operations in half-precision format, while storing minimal information in single-precision to retain as much information as possible in critical parts of the network. Since the introduction of [tensor cores](https://developer.nvidia.com/tensor-cores) in the Volta, and following with both the Turing and Ampere architectures, significant training speedups are experienced by switching to mixed precision -- up to 3x overall speedup on the most arithmetically intense model architectures. Using mixed precision training requires two steps:
 
1.  Porting the model to use the FP16 data type where appropriate.
    
2.  Adding loss scaling to preserve small gradient values.
    
 
  
 
For information about:
 
-   How to train using mixed precision, see the [Mixed Precision Training](https://arxiv.org/abs/1710.03740) paper and [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) documentation.
    
-   Techniques used for mixed precision training, see the [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/) blog.
    
 
-   APEX tools for mixed precision training, see the [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/).
  
 
#### Enabling mixed precision
 
In this repository, mixed precision training is enabled by NVIDIA’s [APEX](https://github.com/NVIDIA/apex) library. The APEX library has an automatic mixed precision module that allows mixed precision to be enabled with minimal code changes.
 
Automatic mixed precision can be enabled with the following code changes: 
 
```
from apex import amp
if fp16:
    # Wrap optimizer and model
    model, optimizer = amp.initialize(model, optimizer, opt_level=<opt_level>, loss_scale="dynamic")
 
if fp16:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
   ```
 
Where <opt_level> is the optimization level. In the MaskRCNN, "O1" is set as the optimization level. Mixed precision training can be turned on by passing in the argument fp16 to the pre-training and fine-tuning Python scripts. Shell scripts all have a positional argument available to enable mixed precision training.
 
#### Enabling TF32
 
 
 
TensorFloat-32 (TF32) is the new math mode in [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for handling the matrix math also called tensor operations. TF32 running on Tensor Cores in A100 GPUs can provide up to 10x speedups compared to single-precision floating-point math (FP32) on Volta GPUs. 
 
TF32 Tensor Cores can speed up networks using FP32, typically with no loss of accuracy. It is more robust than FP16 for models which require high dynamic range for weights or activations.
 
For more information, refer to the [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) blog post.
 
TF32 is supported in the NVIDIA Ampere GPU architecture and is enabled by default.
 
 
## Setup
The following sections list the requirements in order to start training the Mask R-CNN model.
 
### Requirements
 
This repository contains `Dockerfile` which extends the PyTorch NGC container and encapsulates some dependencies.  Aside from these dependencies, ensure you have the following components:
  - [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
  - [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
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
To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the Mask R-CNN model on the COCO 2014 dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.
 
 
### 1. Clone the repository.
```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/Segmentation/MaskRCNN
```
 
### 2. Download and preprocess the dataset.
This repository provides scripts to download and extract the COCO 2014 dataset.  Data will be downloaded to the `current working` directory on the host and extracted to a user-defined directory
 
To download, verify, and extract the COCO dataset, use the following scripts:
  ```
  ./download_dataset.sh <data/dir>
  ```
By default, the data is organized into the following structure:
  ```
  <data/dir>
    annotations/
      instances_train2014.json
      instances_val2014.json
    train2014/
      COCO_train2014_*.jpg
    val2014/
      COCO_val2014_*.jpg
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
 
  The `scripts/train.sh` script runs the following Python command:
  ```
  Volta:
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file “configs/e2e_mask_rcnn_R_50_FPN_1x.yaml”
  
  Ampere:
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file “configs/e2e_mask_rcnn_R_50_FPN_1x_ampere_bs64.yaml”   
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
    OUTPUT_DIR RESULTS \
    SOLVER.BASE_LR 0.002 \
    SOLVER.STEPS ‘(360000, 480000)’
```
  
### Command-line options
 
To see the full list of available options and their descriptions, use the -h or --help command line option, for example:
 
  
 
`python tools/train_net.py --help`
 
 
### Getting the data
The Mask R-CNN model was trained on the [COCO 2014](http://cocodataset.org/#download) dataset.  This dataset comes with a training and validation set.  
 
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
  - Dataset - COCO 2014
  - Base Learning Rate - 0.02
  - Global train batch size - 16
  - Global test batch size - 8
  - RPN batch size - 256
  - ROI batch size - 512
  - Solver steps - (60000, 80000)
  - Max iterations - 90000
  - Warmup iterations - 500
  - Warmup factor = 0.33
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
 
We have tested batch sizes upto 4 on a 16GB V100 and upto 8 on a 32G V100 with mixed precision. The repository also implements gradient accumulation functionality to simulate bigger batches. The following command can be used to run a batch of 64:
  ```
  python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file \"configs/e2e_mask_rcnn_R_50_FPN_1x.yaml\" SOLVER.ACCUMULATE_GRAD True SOLVER.ACCUMULATE_STEPS 4
  ```
 
By default, training is performed using FP32 on Volta and TF32 on Ampere, however training time can be reduced further using tensor cores and mixed precision. This can be done by either adding `--amp` to the command line or `DTYPE \"float16\"` to override the respective parameter in the config file.
 
__Note__: When training a global batch size >= 32, it is recommended to additionally set the following parameters:
  - `SOLVER.WARMUP_ITERS 625`
  - `SOLVER.WARMUP_FACTOR 0.01`
 
When experimenting with different global batch sizes for training and inference, make sure `SOLVER.IMS_PER_BATCH` and `TEST.IMS_PER_BATCH` are divisible by the number of GPUs.  
 
#### Other training options
A sample single GPU config is provided under `configs/e2e_mask_rcnn_R_50_FPN_1x_1GPU.yaml`
 
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
scripts/train_benchmark.sh <float16/tf32/float32>
```
 
#### Inference performance benchmark
Inference benchmarking can be performed by running the script:
```
scripts/inference_benchmark.sh <float16/tf32/float32>
```
 
### Results
The following sections provide details on how we achieved our performance and accuracy in training and inference.
#### Training Accuracy Results
 
##### Training accuracy: NVIDIA DGX A100 (8x A100 40GB)
 
Our results were obtained by running the `scripts/train.sh` training script in the 20.06-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs.
 
| GPUs    | Batch size / GPU    | BBOX mAP - TF32| MASK mAP - TF32  | BBOX mAP - FP16| MASK mAP - FP16  |   Time to train - TF32  |  Time to train - mixed precision | Time to train speedup (TF32 to mixed precision)
| --| --| --  | -- | -- | -- | -- | -- | --
| 8 | 8 | 0.377 | 0.3422 | 0.377 | 0.3424 | 3.63 | 3.37 | 1.077
 
##### Training accuracy: NVIDIA DGX-1 (8x V100 16GB)
 
Our results were obtained by running the `scripts/train.sh`  training script in the PyTorch 20.06-py3 NGC container on NVIDIA DGX-1 with 8x V100 16GB GPUs.
 
| GPUs    | Batch size / GPU    | BBOX mAP - FP32| MASK mAP - FP32  | BBOX mAP - FP16| MASK mAP - FP16  |   Time to train - FP32  |  Time to train - mixed precision | Time to train speedup (FP32 to mixed precision)
| --| --| --  | -- | -- | -- | -- | -- | --
| 8 | 4 | 0.377 | 0.3422 | 0.3767 | 0.3421 | 5.69 | 4.48 | 1.27
 
 
##### Training loss curves
 
![Loss Curve](./img/loss_curve.png)
 
Here, multihead loss is simply the summation of losses on the mask head and the bounding box head.
 
 
##### Training Stability Test
The following tables compare mAP scores across 5 different training runs with different seeds.  The runs showcase consistent convergence on all 5 seeds with very little deviation.
 
| **Config** | **Seed 1** | **Seed 2** | **Seed 3** |  **Seed 4** | **Seed 5** | **Mean** | **Standard Deviation** |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- |
|  8 GPUs, final AP BBox  | 0.377 | 0.376 | 0.376 | 0.378  | 0.377 | 0.377 | 0.001 |
| 8 GPUs, final AP Segm | 0.343 | 0.342 | 0.341 | 0.343  | 0.343 | 0.342 | 0.001 |
 
#### Training Performance Results
 
##### Training performance: NVIDIA DGX A100 (8x A100 40GB)
 
Our results were obtained by running the `scripts/train_benchmark.sh` training script in the 20.06-py3 NGC container on NVIDIA DGX A100 (8x A100 40GB) GPUs. Performance numbers in images per second were averaged over an entire training epoch.
 
| GPUs   | Batch size / GPU   | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)   | Weak scaling - TF32    | Weak scaling - mixed precision
 --- | --- | ----- | ----- | --- | --- | ----- |
| 1 | 8 | 21 | 25 | 1.19 | 1 | 1 |
| 4 | 8 | 74 | 84 | 1.14 | 3.52 | 3.36 |
| 8 | 8 | 145 | 161 | 1.11 | 6.90 | 6.44 |
 
##### Training performance: NVIDIA DGX-1 (8x V100 16GB)
 
Our results were obtained by running the `scripts/train_benchmark.sh` training script in the 20.06-py3 NGC container on NVIDIA DGX-1 with (8x V100 16GB) GPUs. Performance numbers in images per second were averaged over an entire training epoch.
 
| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision
| --- | --- | ----- | ----- | --- | --- | -----
| 1 | 4 | 12 | 15 | 1.25 | 1 | 1 |
| 4 | 4 | 38 | 46 | 1.21 | 3.7 | 3.07 |
| 8 | 4 | 70 | 89 | 1.27 | 5.83 | 5.93 |
 
##### Training performance: NVIDIA DGX-2 (16x V100 32GB)
 
Our results were obtained by running the `scripts/train_benchmark.sh` training script in the 20.06-py3 NGC container on NVIDIA DGX-2 with (16x V100 32GB) GPUs. Performance numbers in images per second were averaged over an entire training epoch.
 
| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)   | Weak scaling - FP32    | Weak scaling - mixed precision        
| --- | --- | ----- | ----- | --- | --- | ----- |
| 1 | 4 | 12 | 16 | 1.33 | 1 | 1 |
| 4 | 4 | 39 | 50 | 1.28 | 3.25 | 3.13 |
| 8 | 4 | 75 | 92 | 1.27 | 6.25 | 5.75 |
| 16 | 4 | 148 | 218 | 1.47 | 12.33 | 13.63 |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
#### Inference performance results
 
##### Inference performance: NVIDIA DGX A100 (1x A100 40GB)
 
Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the PyTorch 20.06-py3 NGC container on NVIDIA DGX A100 (1x A100 40GB) GPU.
 
| GPUs   | Batch size / GPU   | Throughput - TF32    | Throughput - mixed precision    | Throughput speedup (TF32 - mixed precision)
| --- | --- | ----- | ----- | ----- |
|  1  | 8 | 27 | 26 | 0.963 |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).

##### Inference performance: NVIDIA DGX-1 (1x V100 16GB)
 
Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the PyTorch 20.06-py3 NGC container on NVIDIA DGX-1 with 1x V100 16GB GPUs. 
| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)
| --- | --- | ----- | ----- | ----- |
|  1  | 8 | 16 | 19 | 1.188 |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
##### Inference performance: NVIDIA DGX-2 (1x V100 32GB)
 
Our results were obtained by running the `scripts/inference_benchmark.sh` training script in the PyTorch 20.06-py3 NGC container on NVIDIA DGX-1 with 1x V100 32GB GPUs. Performance numbers (in items/images per second) were averaged over an entire training epoch.
 
| GPUs   | Batch size / GPU   | Throughput - FP32    | Throughput - mixed precision    | Throughput speedup (FP32 - mixed precision)
| --- | --- | ----- | ----- | ----- |
|  1  | 8 | 19 | 21 | 1.105 |
 
To achieve these same results, follow the steps in the [Quick Start Guide](#quick-start-guide).
 
## Release notes
 
### Changelog
 
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
There are no known issues with this model.
