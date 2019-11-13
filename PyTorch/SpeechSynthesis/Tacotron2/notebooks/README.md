# Tacotron2 and WaveGlow

A jupyter notobook based on Quick Start Guide of: https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2

## Requirements

Ensure you have the following components:

NVIDIA Docker (https://github.com/NVIDIA/nvidia-docker) PyTorch 19.06-py3+ NGC container or newer (https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) NVIDIA Volta (https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or Turing (https://www.nvidia.com/en-us/geforce/turing/) based GPU

Before running the Jupyter notebook, please make sure you already git clone the code from the Github:


```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git 
    
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
```

Copy the Tacotron2.ipynb file into the folder 'Tacotron2'

```bash
cp notebooks/Tacotron2.ipynb .
```

### Running the quick start guide as a Jupyter notebook

To run the notebook on you local machine:

```bash
jupyter notebook Tacotron2.ipynb
```

To run the notebook remotely:

```bash
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine at port `8888`:

```
http://[host machine]:8888
```

Use the token listed in the output from running the `jupyter` command to log in, for example:

```
http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b
```