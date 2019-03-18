## Model Zoo and Baselines

### Hardware
- 8 NVIDIA V100 GPUs

### Software
- PyTorch version: 1.0.0a0+dd2c487
- CUDA 9.2
- CUDNN 7.1
- NCCL 2.2.13-1

### End-to-end Faster and Mask R-CNN baselines

All the baselines were trained using the exact same experimental setup as in Detectron.
We initialize the detection models with ImageNet weights from Caffe2, the same as used by Detectron.

The pre-trained models are available in the link in the model id.

backbone | type | lr sched | im / gpu | train mem(GB) | train time (s/iter) | total train time(hr) | inference time(s/im) | box AP | mask AP | model id
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --
R-50-C4 | Fast | 1x | 1 | 5.8 | 0.4036 | 20.2 | 0.17130 | 34.8 | - | [6358800](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_C4_1x.pth)
R-50-FPN | Fast | 1x | 2 | 4.4 | 0.3530 | 8.8 | 0.12580 | 36.8 | - | [6358793](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth)
R-101-FPN | Fast | 1x | 2 | 7.1 | 0.4591 | 11.5 | 0.143149 | 39.1 | - | [6358804](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_101_FPN_1x.pth)
X-101-32x8d-FPN | Fast | 1x | 1 | 7.6 | 0.7007 | 35.0 | 0.209965 | 41.2 | - | [6358717](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth)
R-50-C4 | Mask | 1x | 1 | 5.8 | 0.4520 | 22.6 | 0.17796 + 0.028 | 35.6 | 31.5 | [6358801](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_C4_1x.pth)
R-50-FPN | Mask | 1x | 2 | 5.2 | 0.4536 | 11.3 | 0.12966 + 0.034 | 37.8 | 34.2 | [6358792](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth)
R-101-FPN | Mask | 1x | 2 | 7.9 | 0.5665 | 14.2 | 0.15384 + 0.034 | 40.1 | 36.1 | [6358805](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_101_FPN_1x.pth)
X-101-32x8d-FPN | Mask | 1x | 1 | 7.8 | 0.7562 | 37.8 | 0.21739 + 0.034 | 42.2 | 37.8 | [6358718](https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth)


## Comparison with Detectron and mmdetection

In the following section, we compare our implementation with [Detectron](https://github.com/facebookresearch/Detectron)
and [mmdetection](https://github.com/open-mmlab/mmdetection).
The same remarks from [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md#training-speed)
about different hardware applies here.

### Training speed

The numbers here are in seconds / iteration. The lower, the better.

type | Detectron (P100) | mmdetection (V100) | maskrcnn_benchmark (V100)
-- | -- | -- | --
Faster R-CNN R-50 C4 | 0.566 | - | 0.4036
Faster R-CNN R-50 FPN | 0.544 | 0.554 | 0.3530
Faster R-CNN R-101 FPN | 0.647 | - | 0.4591
Faster R-CNN X-101-32x8d FPN | 0.799 | - | 0.7007
Mask R-CNN R-50 C4 | 0.620 | - | 0.4520
Mask R-CNN R-50 FPN | 0.889 | 0.690 | 0.4536
Mask R-CNN R-101 FPN | 1.008 | - | 0.5665
Mask R-CNN X-101-32x8d FPN | 0.961 | - | 0.7562

### Training memory

The lower, the better

type | Detectron (P100) | mmdetection (V100) | maskrcnn_benchmark (V100)
-- | -- | -- | --
Faster R-CNN R-50 C4 | 6.3 | - | 5.8
Faster R-CNN R-50 FPN | 7.2 | 4.9 | 4.4
Faster R-CNN R-101 FPN | 8.9 | - | 7.1
Faster R-CNN X-101-32x8d FPN | 7.0 | - | 7.6
Mask R-CNN R-50 C4 | 6.6 | - | 5.8
Mask R-CNN R-50 FPN | 8.6 | 5.9 | 5.2
Mask R-CNN R-101 FPN | 10.2 | - | 7.9
Mask R-CNN X-101-32x8d FPN | 7.7 | - | 7.8

### Accuracy

The higher, the better

type | Detectron (P100) | mmdetection (V100) | maskrcnn_benchmark (V100)
-- | -- | -- | --
Faster R-CNN R-50 C4 | 34.8 | - | 34.8
Faster R-CNN R-50 FPN | 36.7 | 36.7 | 36.8
Faster R-CNN R-101 FPN | 39.4 | - | 39.1
Faster R-CNN X-101-32x8d FPN | 41.3 | - | 41.2
Mask R-CNN R-50 C4 | 35.8 & 31.4 | - | 35.6 & 31.5
Mask R-CNN R-50 FPN | 37.7 & 33.9 | 37.5 & 34.4 | 37.8 & 34.2
Mask R-CNN R-101 FPN | 40.0 & 35.9 | - | 40.1 & 36.1
Mask R-CNN X-101-32x8d FPN | 42.1 & 37.3 | - | 42.2 & 37.8

