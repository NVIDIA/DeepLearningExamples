Our code base support two types of data loaders.

- [Tensorflow Sequence Generator](#tensorflow-sequence-generator)
- [NVIDIA DALI Generator](#nvidia-dali-generator)

## [Tensorflow Sequence Generator](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence)

Sequence data generator is best suited for situations where we need
advanced control over sample generation or when simple data does not
fit into memory and must be loaded dynamically.

Our [sequence generator](./../data_generators/tf_data_generator.py) generates
dataset on multiple cores in real time and feed it right away to deep
learning model.

## [NVIDIA DALI Generator](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)

The NVIDIA Data Loading Library (DALI) is a library for data loading and
pre-processing to accelerate deep learning applications. It provides a
collection of highly optimized building blocks for loading and processing
image, video and audio data. It can be used as a portable drop-in
replacement for built in data loaders and data iterators in popular deep
learning frameworks.

We've used [DALI Pipeline](./../data_generators/dali_data_generator.py) to directly load
data on `GPU`, which resulted in reduced latency and training time,
mitigating bottlenecks, by overlapping training and pre-processing. Our code
base also support's multi GPU data loading for DALI.

## Use Cases

For training and evaluation you can use both  `TF Sequence` and `DALI` generator with multiple gpus, but for prediction
and inference benchmark we only support `TF Sequence` generator with single gpu support.

> Reminder: DALI is only supported on Linux platforms. For Windows, you can
> train using Sequence Generator. The code base will work without DALI
> installation too.

It's advised to use DALI only when you have large gpu memory to load both model
and training data at the same time.

Override `DATA_GENERATOR_TYPE` in config to change default generator type. Possible
options are `TF_GENERATOR` for Sequence generator and `DALI_GENERATOR` for DALI generator.
 
