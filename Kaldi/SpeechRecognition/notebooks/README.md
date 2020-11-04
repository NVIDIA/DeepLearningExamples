```
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
```
<img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">

# Kaldi  inference demo

## 1. Overview

This folder contains two notebooks demonstrating the steps for carrying out inferencing with the Kaldi TRTIS backend server using a Python gRPC client.
 
- [Offline](Kaldi_TRTIS_inference_offline_demo.ipynb): we will stream pre-recorded .wav files to the inference server and receive the results back.
- [Online](Kaldi_TRTIS_inference_online_demo.ipynb): we will stream live audio stream from a microphone to the inference server and receive the results back.

## 2. Quick Start Guide

First, clone the repository:

```
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/Kaldi/SpeechRecognition
```
Next, build the NVIDIA Kaldi TRTIS container:

```
scripts/docker/build.sh
```

Then download the model and some test data set with:
```
scripts/docker/launch_download.sh
```
Next, launch the TRTIS container with:
```
scripts/docker/launch_server.sh
```
After this step, we should have a TRTIS server ready to serve ASR inference requests.

The next step is to build a TRTIS client container:

```bash
docker build -t kaldi_notebook_client -f Dockerfile.notebook .
```

Start the client container with:

```bash
docker run -it --rm --net=host --device /dev/snd:/dev/snd -v $PWD:/Kaldi kaldi_notebook_client
```

Within the client container, start Jupyter notebook server:

```bash
cd /Kaldi
jupyter notebook --ip=0.0.0.0 --allow-root
```

And navigate a web browser to the IP address or hostname of the host machine
at port `8888`:

```
http://[host machine]:8888
```

Use the token listed in the output from running the `jupyter` command to log
in, for example:

```
http://[host machine]:8888/?token=aae96ae9387cd28151868fee318c3b3581a2d794f3b25c6b
```
