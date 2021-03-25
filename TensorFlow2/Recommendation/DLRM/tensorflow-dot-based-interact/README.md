# TF 2.x Dot Based Interacti CUDA Op

## Requirements

This op needs to run from within a TF2 NGC containter >= 20.12. E.g.:
```
docker pull gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:21.02-tf2-py3-devel

docker run -it [...] gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:21.02-tf2-py3-devel
```

## Installation

The package with built binaries will be internally hosted on [my PyPi package registry](https://gitlab-master.nvidia.com/wraveane/pypi). There are two ways to install this:

- Either: Install directly via PIP using my gitlab token (or replace the URL with your own token):
  ```
  pip3 install --extra-index-url https://__token__:TmAosCzLDiFzS7x3J1aN@gitlab-master.nvidia.com/api/v4/projects/38036/packages/pypi/simple tensorflow-dot-based-interact
  ```
- Or: Manually download the wheel package file from the [package's registry page](https://gitlab-master.nvidia.com/wraveane/pypi/-/packages/1376), and install it:
  ```
  pip install ./tensorflow_dot_based_interact-*.whl
  ```

## Build from Source

Alternatively, it can be built from source as follows:

- Fix the TF CUDA include directory:
  ```
  mkdir -p /usr/local/lib/python3.8/dist-packages/tensorflow/include/third_party/gpus/cuda/
  ln -s /usr/local/cuda/include /usr/local/lib/python3.8/dist-packages/tensorflow/include/third_party/gpus/cuda/
  ```
- Clone this repository and build it:
  ```
  git clone https://gitlab-master.nvidia.com/wraveane/tensorflow-dot-based-interact
  cd tensorflow-dot-based-interact
  make
  ```
- Run the [unit tests](tensorflow_dot_based_interact/python/ops/dot_based_interact_ops_test.py) to ensure the op is working as intended:
  ```
  make test
  ```
- Install the TF Op package in one of two ways:
  - Either: Create a wheel and install it with pip:
    ```
    make pkg
    pip install ./artifacts/tensorflow_dot_based_interact-*.whl
    ```
  - Or: Install the repository directory locally:
    ```
    pip install -e .
    ```

## Usage

The architecture to be used is as follows:

![Dot Based Interact](https://docs.google.com/drawings/d/e/2PACX-1vT-RW1_SsvfENGogMxiqM8_pwDR6m8WXklWzX5kICDOJLK_0XPfO2oLyo_G9apVDXsc9LYE2XP7_e9I/pub?w=368&h=489)

Where the TF CUDA op implemented by this package takes two inputs:
- **input**: The concatenation (done in TensorFlow) of the Bottom MLP output and the embeddings.
- **bottom_mlp_output**: A copy of the Bottom MLP output tensor.

The result of the operation will already have the Bottom MLP output tensor concatenated, ready to be given to the next stage of the architecture.

To use it, follow the installation or building instructions for the package above. Then:

- Make sure the op is properly installed:
  ```
  pip show tensorflow-dot-based-interact
  ```
- Use it like this:
  ```
  from tensorflow_dot_based_interact.python.ops import dot_based_interact_ops

  bottom_mlp_output = ...   # The bottom MLP output tensor
  embeddings = ...          # The sparse features embeddings tensor

  input = tf.concat([bottom_mlp_output, embeddings], axis=1)   # Bottom Concat

  result = dot_based_interact_ops.dot_based_interact(input, bottom_mlp_output)
  ```

## Support

The TF DBI custom op will dynamically switch kernel versions according to:

- GPU Architecture:
  - GPU Major Version >= 8: Ampere Kernels
  - GPU Major Version == 7: Volta Kernels
  - GPU Major Version <= 6: Not Supported / Error Thrown
- Data Alignment
- Data Type:
  - Ampere:
    - TF32 (on aligned inputs)
    - FP32 (fallback on non-aligned inputs)
    - FP16
  - Volta:
    - FP32
    - FP16
