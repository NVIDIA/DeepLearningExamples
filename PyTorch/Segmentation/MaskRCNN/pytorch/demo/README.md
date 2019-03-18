## Webcam and Jupyter notebook demo

This folder contains a simple webcam demo that illustrates how you can use `maskrcnn_benchmark` for inference.


### With your preferred environment

You can start it by running it from this folder, using one of the following commands:
```bash
# by default, it runs on the GPU
# for best results, use min-image-size 800
python webcam.py --min-image-size 800
# can also run it on the CPU
python webcam.py --min-image-size 300 MODEL.DEVICE cpu
# or change the model that you want to use
python webcam.py --config-file ../configs/caffe2/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml --min-image-size 300 MODEL.DEVICE cpu
# in order to see the probability heatmaps, pass --show-mask-heatmaps
python webcam.py --min-image-size 300 --show-mask-heatmaps MODEL.DEVICE cpu
```

### With Docker

Build the image with the tag `maskrcnn-benchmark` (check [INSTALL.md](../INSTALL.md) for instructions)

Adjust permissions of the X server host (be careful with this step, refer to 
[here](http://wiki.ros.org/docker/Tutorials/GUI) for alternatives)

```bash
xhost +
``` 

Then run a container with the demo:
 
```
docker run --rm -it \
    -e DISPLAY=${DISPLAY} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device=/dev/video0:/dev/video0 \
    --ipc=host maskrcnn-benchmark \
    python demo/webcam.py --min-image-size 300
```

**DISCLAIMER:** *This was tested for an Ubuntu 16.04 machine, 
the volume mapping may vary depending on your platform*
