
# Convolutional neural network training scripts

This script implements a number of popular CNN models and demonstrates
efficient single-node training on multi-GPU systems. It can be used for
benchmarking, training and evaluation of models.

Uber's Horovod data-parallel framework is used for parallelization.


## Imagenet data preprocessing

See [this file](data_preprocessing/README.md) for instructions on downloading
and preprocessing the imagenet data set.


## ResNet50 training example

The following command initiates training of the ResNet50 model distributed
across 8 GPUs using fp16 arithmetic. We assume Imagenet is saved in TFRecord
format at /data/imagenet_tfrecord.

```
    $ mpiexec -np 8 python nvcnn_hvd.py \
                      --model=resnet50 \
                      --data_dir=/data/imagenet_tfrecord \
                      --batch_size=256 \
                      --fp16 \
                      --larc_mode=clip \
                      --larc_eta=0.003 \
                      --loss_scale=128 \
                      --log_dir=./checkpoint-dir \
                      --save_interval=3600 \
                      --num_epochs=90 \
                      --display_every=100
                      --learning_rate=2.0
```


## Inception V3 training example

The following command initiates training of the Inception V3 model distributed
across 8 GPUs using fp16 arithmetic. We assume Imagenet is saved in TFRecord
format at /data/imagenet_tfrecord.

```
    $ mpiexec -np 8 python nvcnn_hvd.py \
                      --model=inception3 \
                      --data_dir=/data/imagenet_tfrecord \
                      --batch_size=128 \
                      --fp16 \
                      --larc_mode=clip \
                      --larc_eta=0.003 \
                      --loss_scale=128 \
                      --log_dir=./checkpoint-dir \
                      --save_interval=3600 \
                      --num_epochs=90 \
                      --display_every=100 \
                      --learning_rate=1.0
```


## Evaluating accuracy with the test set

Model parameters are stored in FP32 precision when training with either FP32 or
FP16 arithmetic. Thus the `--fp16` flag is not needed for eval jobs. Also,
evaluation is performed on a single GPU. The following command performs
evaluation of a trained model.

```
    $ python nvcnn_hvd.py --model=<resnet50|inception3> \
                          --data_dir=/data/imagenet_tfrecord \
                          --batch_size=256 \
                          --log_dir=./checkpoint-dir \
                          --eval
```

After trianing, ResNet50 and Inception V3 should achieve top-1 accuracies of
75.5% and 77.8%, respectively, on the imagenet validation set.


##  Notes

With the `--fp16` flag the model is trained using 16-bit floating-point
operations. This provides optimized performance on Volta's TensorCores.
For more information on training with FP16 arithmetic see
[Training with Mixed Precision](http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html).

If executing the training command above as root (for example in a Docker
container), mpiexec requires an additional --allow-run-as-root flag.

