
# Tacotron 2 and WaveGlow inference on TRTIS

## Setup

### Clone the repository.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
```

### Obtain models to be loaded in TRTIS.

We have prepared Tacotron 2 and WaveGlow models that are ready to be loaded in TRTIS,
so you don't need to train and export the models. Please follow the instructions 
below to learn how to train, export --- or simply download the pretrained models. 

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
python -m multiproc train.py -m WaveGlow -o output/ --amp-run -lr 1e-4 --epochs 2001 --wn-channels 256 -bs 12 --segment-length 16000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-benchmark --cudnn-enabled --log-file output/nvlog.json
```

This will train the WaveGlow model with a smaller number of residual connections
in the coupling layer networks and larger segment length. Training should take 
about 100 hours on DGX-1 (8x V100 16G).

### Setup Tacotron 2 TorchScript.

First, you need to create a folder structure for the model to be loaded in TRTIS server.
Follow the Tacotron 2 Quick Start Guide (points 1-4) to start the container.
Inside the container, type:
```bash
cd /workspace/tacotron2/
python exports/export_tacotron2_ts_config.py --amp-run
```

This will export the folder structure of the TRTIS repository and the config file of Tacotron 2. 
By default, it will be found in the `trtis_repo/tacotron2` folder.

Now there are two ways to proceed.

#### (Option 1) Download the Tacotron 2 TorchScript model.

Download the Tacotron 2 TorchScript model from:
- [Tacotron2 TorchScript](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_jit_fp16)

Move the downloaded model to `trtis_repo/tacotron2/1/model.pt`

#### (Option 2) Export the Tacotron 2 model using TorchScript.

To export the Tacotron 2 model using TorchScript, type:
```bash
python exports/export_tacotron2_ts.py --tacotron2 <tacotron2_checkpoint> -o trtis_repo/tacotron2/1/model.pt --amp-run
```

This will save the model as ``trtis_repo/tacotron2/1/model.pt``.

### Setup WaveGlow TRT engine.

For WaveGlow, we also need to create the folder structure that will be used by the TRTIS server. 
Inside the container, type:
```bash
cd /workspace/tacotron2/
python exports/export_waveglow_trt_config.py --amp-run
```

This will export the folder structure of the TRTIS repository and the config file of Waveglow. 
By default, it will be found in the `trtis_repo/waveglow` folder.

There are two ways to proceed. 

#### (Option 1) Download the WaveGlow TRT engine.

Download the WaveGlow TRT engine from:
- [WaveGlow TRT engine](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_trt_fp16)

Move the downloaded model to `trtis_repo/waveglow/1/model.plan`

#### (Option 2) Export the WaveGlow model to TRT.

Before exporting the model, you need to install onnx-tensorrt by typing:
```bash
cd /workspace && git clone https://github.com/onnx/onnx-tensorrt.git
cd /workspace/onnx-tensorrt/ && git submodule update --init --recursive
cd /workspace/onnx-tensorrt && mkdir -p build
cd /workspace/onnx-tensorrt/build && cmake .. -DCMAKE_CXX_FLAGS=-isystem\ /usr/local/cuda/include && make -j12 && make install
```

In order to export the model into the ONNX intermediate representation, type:

```bash
python exports/export_waveglow_onnx.py --waveglow <waveglow_checkpoint> --wn-channels 256 --amp-run --output ./output
```

This will save the model as `waveglow.onnx` (you can change its name with the flag `--output <filename>`).

With the model exported to ONNX, type the following to obtain a TRT engine and save it as `trtis_repo/waveglow/1/model.plan`:

```bash
python trt/export_onnx2trt.py --waveglow  <exported_waveglow_onnx> -o trtis_repo/waveglow/1/ --fp16
```

### Setup the TRTIS server.

Download the TRTIS container by typing:
```bash
docker pull nvcr.io/nvidia/tensorrtserver:20.01-py3
docker tag nvcr.io/nvidia/tensorrtserver:20.01-py3 tensorrtserver:20.01
```

### Setup the TRTIS notebook client.

Now go to the root directory of the Tacotron 2 repo, and type: 

```bash
docker build -f Dockerfile_trtis_client --network=host -t speech_ai_tts_only:demo .
```

### Run the TRTIS server.

To run the server, type in the root directory of the Tacotron 2 repo:
```bash
NV_GPU=1 nvidia-docker run -ti --ipc=host --network=host --rm -p8000:8000 -p8001:8001 -v $PWD/trtis_repo/:/models tensorrtserver:20.01 trtserver --model-store=/models --log-verbose 1
```

The flag `NV_GPU` selects the GPU the server is going to see. If we want it to see all the available GPUs, then run the above command without this flag.
By default, the model repository will be in `trtis_repo/`.

### Run the TRTIS notebook client.

Leave the server running. In another terminal, type:
```bash
docker run -it --rm --network=host --device /dev/snd:/dev/snd --device /dev/usb:/dev/usb speech_ai_tts_only:demo bash ./run_this.sh
```

Open the URL in a browser, open `notebook.ipynb`, click play, and enjoy.
