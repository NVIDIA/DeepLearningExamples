<!-- #region -->
# DLRM Jupyter demo notebooks

This folder contains the demo notebooks for DLRM. The most convenient way to use these notebooks is via using a docker container, which provides a self-contained, isolated and re-producible environment for all experiments. Refer to the [Quick Start Guide section](../README.md) of the Readme documentation for a comprehensive guide. 

First, clone the repository:

```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/PyTorch/Recommendation/DLRM
```


## Notebook list

### 1. Pytorch_DLRM_pyt_train_and_inference.ipynb: training and inference demo

To execute this notebook, first build the DLRM container:
```
docker build . -t nvidia_dlrm_pyt
```

Make a directory for storing DLRM data and start a docker containerexport PYTHONPATH=/workspace/dlrm with:
```
mkdir -p data
docker run --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data nvidia_dlrm_pyt bash
```

Within the docker interactive bash session, start Jupyter with

```
export PYTHONPATH=/workspace/dlrm
jupyter notebook --ip 0.0.0.0 --port 8888
```

Then open the Jupyter GUI interface on your host machine at http://localhost:8888. Within the container, this demo notebook is located at `/workspace/dlrm/notebooks`.
<!-- #endregion -->

### 2. DLRM_Triton_inference_demo.ipynb: inference demo with the NVIDIA Triton Inference server.

To execute this notebook, first build the following inference container:

```
docker build -t dlrm-inference . -f triton/Dockerfile
```

Start in interactive docker session with:

```
docker run -it --rm --gpus device=0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v <PATH_TO_SAVED_MODEL>:/models -v <PATH_TO_EXPORT_MODEL>:/repository dlrm-inference bash
```
where:

- PATH_TO_SAVED_MODEL: directory containing the trained DLRM models.
 
- PATH_TO_EXPORT_MODEL: directory which will contain the converted model to be used with the NVIDIA Triton inference server.

Within the docker interactive bash session, start Jupyter with

```
export PYTHONPATH=/workspace/dlrm
jupyter notebook --ip 0.0.0.0 --port 8888
```

Then open the Jupyter GUI interface on your host machine at http://localhost:8888. Within the container, this demo notebook is located at `/workspace/dlrm/notebooks`.

```python

```
