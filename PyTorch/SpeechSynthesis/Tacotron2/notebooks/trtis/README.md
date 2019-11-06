## Clone the repository.
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
```

## Obtain models to be loaded in TRTIS.

We prepared Tacotron 2 and WaveGlow models that are ready to be loaded in TRTIS.
If you want to create your own models, please follow the instructions on how to 
train and export the models below.

## Obtain Tacotron 2 and WaveGlow checkpoints.

You can either download pretrained models or train the models yourself. Both
options are described in the following sections.

### Download pretrained checkpoints.

Simply download checkpoints from:

### Train Tacotron 2 and WaveGlow models.

Follow the QuickStart section in the `Tacotron2/README.md` file by executing
points 1-5 in the Docker container. To train WaveGlow, use the following command
instead of the one given in QuickStart point 5:

```bash
python -m multiproc train.py -m WaveGlow -o output/ --amp-run -lr 1e-4 --epochs 2001 --wn-channels 256 -bs 12 --segment-length 16000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-benchmark --cudnn-enabled --log-file output/nvlog.json
```

This will train the WaveGlow model with smaller number of residual connections
and larger segment length. Training should take about 100 hours.

## Export Tacotron 2 model using TorchScript

Start the Tacotron 2 docker container. 
Inside the container, from the model root directory type:
```bash
python export_tacotron2_ts_config.py --amp-run
```
This will export the folder structure of the TRTIS repository and the config file of Tacotron 2. By default, it will be found in the `trtis_repo/tacotron` folder.

Now type:
```bash
python export_tacotron2_ts.py --tacotron2 <tacotron2_checkpoint> -o trtis_repo/tacotron2/1/model.pt --amp-run
```

This will save the model as ``trtis_repo/tacotron/1/model.pt``.


## Export WaveGlow model to TRT

Before exporting the model, you need to install onnx-tensorrt by typing:
```bash
cd /workspace && git clone https://github.com/onnx/onnx-tensorrt.git
cd /workspace/onnx-tensorrt/ && git submodule update --init --recursive
cd /workspace/onnx-tensorrt && mkdir -p build
cd /workspace/onnx-tensorrt/build && cmake .. -DCMAKE_CXX_FLAGS=-isystem\ /usr/local/cuda/include && make -j12 && make install
```

Now, type:
```bash
cd /workspace/tacotron2/
python export_waveglow_trt_config.py --amp-run
```

This will export the folder structure of the TRTIS repository and the config file of Waveglow. By default, it will be found in the `trtis_repo/waveglow` folder.

In order to export the model into the ONNX intermediate format, type:

```bash
python export_waveglow_trt.py --waveglow <waveglow_checkpoint> --wn-channels 256 --amp-run
```

This will save the model as `waveglow.onnx` (you can change its name with the flag `--output <filename>`).

With the model exported to ONNX, type the following to obtain a TRT engine and save it as `trtis_repo/waveglow/1/model.plan`:

```bash
onnx2trt <exported_waveglow_onnx> -o trtis_repo/waveglow/1/model.plan -b 1 -w 8589934592
```
Save the folder structure under `trtis_repo` and its contents into the Tacotron 2 repo outside the container. Now exit the Tacotron 2 container.

## Setting up the TRTIS server

```bash
docker pull nvcr.io/nvidia/tensorrtserver:19.10-py3
docker tag nvcr.io/nvidia/tensorrtserver:19.10-py3 tensorrtserver:19.10
```

## Setting up the TRTIS notebook client

Now go to the root directory of the Tacotron 2 repo, and type: 

```bash
docker build -f Dockerfile_trtis_client --network=host -t speech_ai__tts_only:demo .
```

## Running the TRTIS server

```bash
NV_GPU=1 nvidia-docker run -ti --ipc=host --network=host --rm -p8000:8000 -p8001:8001 -v $PWD/trtis_repo/:/models tensorrtserver:19.10 trtserver --model-store=/models --log-verbose 1
```

The flag `NV_GPU` selects the GPU the server is going to see. If we want it to see all the available GPUs, then run the above command without this flag.
By default, the model repository will be in `$PWD/trtis_repo/`.

## Running the TRTIS notebook client

Leave the server running. In another terminal, type in the Tacotron 2 repo:
```bash
docker run -it --rm --network=host --device /dev/snd:/dev/snd --device /dev/usb:/dev/usb speech_ai__tts_only:demo bash ./run_this.sh
```

Open the URL in a browser, open `notebook.ipynb`, click play, and enjoy.
