## Jupyter demo notebooks
This folder contains demo notebooks for the TensorFlow UNet Industrial model.

### 1. TensorFlow_UNet_Industrial_TF_train_and_inference.ipynb: end to end training and inference demo.

The most convenient way to make use of the NVIDIA Tensorflow UNet model is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments. Refer to the [Quick Start Guide section](https://github.com/vinhngx/DeepLearningExamples/tree/vinhn_unet_industrial_demo/TensorFlow/Segmentation/UNet_Industrial#requirements) of the Readme documentation for a comprehensive guide. We briefly summarize the steps here.

First, clone the repository:

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/TensorFlow/Segmentation/UNet_Industrial
```

Next, build the NVIDIA UNet_Industrial container:

```
docker build . --rm -t unet_industrial:latest
```

Then launch the container with:

```
nvidia-docker run -it --rm \
    --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/dataset:/data/dagm2007/ \
    -v /path/to/results:/results \
    unet_industrial:latest
```
where `/path/to/dataset` is the path on the host machine where the data was/is to be downloaded. More on data set preparation in the next section. `/path/to/results` is wher the trained model will be stored.

Within the docker interactive bash session, start Jupyter with

```
jupyter notebook --ip 0.0.0.0 --port 8888
```

Then open the Jupyter GUI interface on your host machine at http://localhost:8888. Within the container, this notebook itself is located at `/workspace/unet_industrial/notebooks`.

### 2. Colab_UNet_Industrial_TF_TFTRT_inference_demo.ipynb: inference from a pretrained UNet model with TensorFlow-TensorRT (TF-TRT).

This notebook is designed to run on Google Colab via this [link](https://colab.research.google.com/github/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFTRT_inference_demo.ipynb)

### 3. Colab_UNet_Industrial_TF_TFHub_export.ipynb: Colab notebook demostrating creation of TF-Hub module from NVIDIA NGC UNet model.
This notebook is designed to run on Google Colab vie this [link](https://colab.research.google.com/github/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_export.ipynb)

### 4. Colab_UNet_Industrial_TF_TFHub_inference_demo.ipynb: Colab notebook demostrating inference with TF-Hub UNet module.
This notebook is designed to run on Google Colab vie this [link](https://colab.research.google.com/github/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Segmentation/UNet_Industrial/notebooks/Colab_UNet_Industrial_TF_TFHub_inference_demo.ipynb)

