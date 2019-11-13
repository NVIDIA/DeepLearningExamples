## Jupyter demo notebooks
This folder contains demo notebooks for the MaskRCNN model.

1 - pytorch_MaskRCNN_pyt_train_and_inference.ipynb: end to end training and inference demo.

The most convenient way to make use of this notebook is via a docker container, which provides a self-contained, isolated and re-producible environment for all experiments. The steps to follow are:

First, clone the repository:

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/Segmentation/MaskRCNN
```

Next, build the NVIDIA Mask R-CNN container:

```
cd pytorch
docker build --rm -t nvidia_joc_maskrcnn_pt .
```

Then launch the container with:

```
PATH_TO_COCO='/path/to/coco-2014'
MOUNT_LOCATION='/datasets/data'
NAME='nvidia_maskrcnn'

docker run --it --runtime=nvidia -p 8888:8888 -v $PATH_TO_COCO:/$MOUNT_LOCATION --rm --name=$NAME --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host nvidia_joc_maskrcnn_pt
```
where `/path/to/coco-2014` is the path on the host machine where the data was/is to be downloaded.

Within the docker interactive bash session, start Jupyter with

`jupyter notebook --ip 0.0.0.0 --port 8888`

Then open the Jupyter GUI interface on your host machine at http://localhost:8888. Within the container, this notebook itself is located at /workspace/object_detection/notebooks.