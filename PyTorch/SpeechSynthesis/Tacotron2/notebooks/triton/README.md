# Tacotron 2 and WaveGlow inference on Triton Inference Server

## Setup

### Clone the repository.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
```

### Obtain models to be loaded in Triton Inference Server.

We have prepared Tacotron 2 and WaveGlow models that are ready to be loaded in
Triton Inference Server, so you don't need to train and export the models.
Please follow the instructions below to learn how to train,
export --- or simply download the pretrained models.

### Obtain Tacotron 2 and WaveGlow checkpoints.

You can either download the pretrained checkpoints or train the models yourself.

#### (Option 1) Download pretrained checkpoints.

If you want to use a pretrained checkpoints, download them from [NGC](https://ngc.nvidia.com/catalog/models):

- [Tacotron2 checkpoint](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_fp16)
- [WaveGlow checkpoint](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16)


#### (Option 2) Train Tacotron 2 and WaveGlow models.

In order to train the models, follow the QuickStart section in the `Tacotron2/README.md`
file by executing points 1-5. You have to train WaveGlow in a different way than described there. Use
the following command instead of the one given in QuickStart at point 5:

```bash
python -m multiproc train.py -m WaveGlow -o output/ --amp -lr 1e-4 --epochs 2001 --wn-channels 256 -bs 12 --segment-length 16000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-benchmark --cudnn-enabled --log-file output/nvlog.json
```

This will train the WaveGlow model with a smaller number of residual connections
in the coupling layer networks and larger segment length. Training should take
about 100 hours on DGX-1 (8x V100 16G).

### Setup Tacotron 2 TorchScript.

There are two ways to proceed.

#### (Option 1) Download the Tacotron 2 TorchScript model.

Download the Tacotron 2 TorchScript model from:
- [Tacotron2 TorchScript](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_jit_fp16)

Next, save it to `triton_models/tacotron2-ts-script/1/` and rename as `model.pt`:

```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2pyt_jit_fp16/versions/1/files/nvidia_tacotron2pyt_jit_fp16
mkdir -p triton_models/tacotron2-ts-script/1/
mv nvidia_tacotron2pyt_jit_fp16 triton_models/tacotron2-ts-script/1/model.pt
```

Copy the Triton config file for the Tacotron 2 model to the model directory:

```bash
cp notebooks/triton/tacotron2_ts-script_config.pbtxt triton_models/tacotron2-ts-script/config.pbtxt
```

#### (Option 2) Export the Tacotron 2 model using TorchScript.

To export the Tacotron 2 model using TorchScript, type:
```bash
python exports/export_tacotron2.py --triton-model-name tacotron2-ts-script --export ts-script -- --checkpoint <Tacotron 2 checkpoint> --config-file config.json
```
This will create the model as file `model.pt` and save it in folder `triton_models/tacotron2-ts-script/1/`.
The command will also generate the Triton configuration file `config.pbtxt` for the Tacotron 2 model.
You can change the folder names using the flags `--triton-models-dir` (default `triton_models`), `--triton-model-name` (default `""`) and `--triton-model-version` (default `1`).
You can also change model file name with the flag `--export-name <filename>`.

### Setup WaveGlow TensorRT engine.

There are two ways to proceed.

#### (Option 1) Download the WaveGlow TensorRT engine.

Download the WaveGlow TensorRT engine from:
- [WaveGlow TensorRT engine](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_trt_fp16)
Next, save it to `triton_models/waveglow-tensorrt/1/` and rename as `model.plan`:

```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/waveglow256pyt_trt_fp16/versions/1/files/nvidia_waveglow256pyt_trt_fp16
mkdir -p triton_models/waveglow-tensorrt/1/
mv nvidia_waveglow256pyt_trt_fp16 triton_models/waveglow-tensorrt/1/model.plan
```

Copy the Triton config file for the WaveGlow model to the model directory:

```bash
cp notebooks/triton/waveglow_tensorrt_config.pbtxt triton_models/waveglow-tensorrt/config.pbtxt
```

#### (Option 2) Export the WaveGlow model to TensorRT.

In order to export the model into the TensorRT engine, type:

```bash
python exports/export_waveglow.py --triton-model-name waveglow-tensorrt --export tensorrt --tensorrt-fp16 -- --checkpoint <waveglow_checkpoint> --config-file config.json --wn-channels 256
```

This will create the model as file `model.plan` and save it in folder `triton_models/waveglow-tensorrt/1/`.
The command will also generate the Triton configuration file `config.pbtxt` for the WaveGlow model.
You can change the folder names using the flags `--triton-models-dir` (default `triton_models`), `--triton-model-name` (default `""`) and `--triton-model-version` (default `1`).
You can also change model file name with the flag `--export-name <filename>`.

### Setup the Triton Inference Server.

Download the Triton Inference Server container by typing:
```bash
docker pull nvcr.io/nvidia/tritonserver:20.06-py3
docker tag nvcr.io/nvidia/tritonserver:20.06-py3 tritonserver:20.06
```

### Setup the Triton notebook client.

Now go to the root directory of the Tacotron 2 repo, and type:

```bash
docker build -f Dockerfile_triton_client --network=host -t speech_ai_tts_only:demo .
```

### Run the Triton Inference Server.

To run the server, type in the root directory of the Tacotron 2 repo:
```bash
NV_GPU=1 nvidia-docker run -ti --ipc=host --network=host --rm -p8000:8000 -p8001:8001 -v $PWD/triton_models/:/models tritonserver:20.06 tritonserver --model-store=/models --log-verbose 1
```

The flag `NV_GPU` selects the GPU the server is going to see. If we want it to see all the available GPUs, then run the above command without this flag.
By default, the model repository will be in `triton_models/`.

### Run the Triton notebook client.

Leave the server running. In another terminal, type:
```bash
docker run -it --rm --network=host --device /dev/snd:/dev/snd speech_ai_tts_only:demo bash ./run_this.sh
```

Open the URL in a browser, open `notebook.ipynb`, click play, and enjoy.
