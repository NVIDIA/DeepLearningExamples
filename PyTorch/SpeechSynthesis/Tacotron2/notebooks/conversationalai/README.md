
## Model Preparation

### Clone the repository

```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples
```

You will build our ConversationalAI in the Tacotron2 folder:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/notebooks/conversationalai
```

### Download checkpoints

Download the PyTorch checkpoints from [NGC](https://ngc.nvidia.com/models):
* [Jasper](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16/files)

```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/jasperpyt_fp16/versions/1/files/jasper_fp16.pt
```


* [BERT](https://ngc.nvidia.com/catalog/models/nvidia:bert_large_pyt_amp_ckpt_squad_qa1_1/files?version=1)

```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/bert_large_pyt_amp_ckpt_squad_qa1_1/versions/1/files/bert_large_qa.pt
```


* [Tacotron 2](https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_pyt_ckpt_amp/files?version=19.12.0)
```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.12.0/files/nvidia_tacotron2pyt_fp16.pt
```


* [WaveGlow](https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ckpt_amp_256/files?version=20.01.0)
```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/waveglow_ckpt_amp_256/versions/20.01.0/files/nvidia_waveglow256pyt_fp16.pt
```


Move the downloaded checkpoints to `models` directory:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/notebooks/conversationalai
```

### Prepare Jasper

First, let's generate a TensorRT engine for Jasper using TensorRT version 7.

Download the Jasper checkpoint from [NGC](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_fp16/files)
and move it to `Jasper/checkpoints/` direcotry:

```bash
mkdir -p DeepLearningExamples/PyTorch/SpeechRecognition/Jasper/checkpoints
mv jasper_fp16.pt DeepLearningExamples/PyTorch/SpeechRecognition/Jasper/checkpoints
```

Apply a patch to enable support of TensorRT 7:

```bash
cd DeepLearningExamples/
git apply --ignore-space-change --reject --whitespace=fix ../patch_jasper_trt7
```

Now, build a container for Jasper:

```bash
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper/
bash tensorrt/scripts/docker/build.sh
```

To run the container, type:

```bash
cd DeepLearningExamples/PyTorch/SpeechRecognition/Jasper
export JASPER_DIR=${PWD}
export DATA_DIR=$JASPER_DIR/data/
export CHECKPOINT_DIR=$JASPER_DIR/checkpoints/
export RESULT_DIR=$JASPER_DIR/results/
cd $JASPER_DIR
mkdir -p $DATA_DIR $CHECKPOINT_DIR $RESULT_DIR
bash tensorrt/scripts/docker/launch.sh $DATA_DIR $CHECKPOINT_DIR $RESULT_DIR
```

Inside the container export Jasper TensorRT engine by executing:

```bash
pip install --upgrade onnx
mkdir -p /results/onnxs/ /results/engines/
cd /jasper
python tensorrt/perf.py --batch_size 1 --engine_batch_size 1 --model_toml configs/jasper10x5dr_nomask.toml --ckpt_path /checkpoints/jasper_fp16.pt --trt_fp16 --pyt_fp16 --engine_path /results/engines/jasper_fp16.engine --onnx_path /results/onnxs/fp32_DYNAMIC.onnx --seq_len 3600 --make_onnx
```

After successful export, copy the engine to model_repo:

```bash
cd DeepLearningExamples/Pytorch
mkdir -p SpeechSynthesis/Tacotron2/notebooks/conversationalai/model_repo/jasper-trt/1
cp SpeechRecognition/Jasper/results/engines/jasper_fp16.engine SpeechSynthesis/Tacotron2/notebooks/conversationalai/model_repo/jasper-trt/1/
```

You will also need Jasper feature extractor and decoder. Download them from [NGC](https://ngc.nvidia.com/catalog/models/nvidia:jasperpyt_jit_fp16/files) and move to the model_repo:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/notebooks/conversationalai/model_repo/
mkdir -p jasper-decoder/1 jasper-feature-extractor/1
wget -P jasper-decoder/ https://api.ngc.nvidia.com/v2/models/nvidia/jasperpyt_jit_fp16/versions/1/files/jasper-decoder/config.pbtxt
wget -P jasper-decoder/1/ https://api.ngc.nvidia.com/v2/models/nvidia/jasperpyt_jit_fp16/versions/1/files/jasper-decoder/1/jasper-decoder.pt
wget -P jasper-feature-extractor/ https://api.ngc.nvidia.com/v2/models/nvidia/jasperpyt_jit_fp16/versions/1/files/jasper-feature-extractor/config.pbtxt
wget -P jasper-feature-extractor/1/ https://api.ngc.nvidia.com/v2/models/nvidia/jasperpyt_jit_fp16/versions/1/files/jasper-feature-extractor/1/jasper-feature-extractor.pt
```

### Prepare BERT

With the generated Jasper model, we can proceed to BERT.

Download the BERT checkpoint from [NGC](https://ngc.nvidia.com/catalog/models/nvidia:bert_large_pyt_amp_ckpt_squad_qa1_1/files)
and move it to `BERT/checkpoints/` direcotry:

```bash
mkdir -p DeepLearningExamples/PyTorch/LanguageModeling/BERT/checkpoints/
mv bert_large_qa.pt DeepLearningExamples/PyTorch/LanguageModeling/BERT/checkpoints/bert_qa.pt
```

Now, build a container for BERT:

```bash
cd PyTorch/LanguageModeling/BERT/
bash scripts/docker/build.sh
```

Use the Triton export script to convert the model `checkpoints/bert_large_qa.pt` to ONNX:

```bash
bash triton/export_model.sh
```

The model will be saved in `results/triton_models/bertQA-onnx`, together with Triton configuration file. Copy the model and configuration file to the model_repo:

```bash
cd DeepLearningExamples
cp -r PyTorch/LanguageModeling/BERT/results/triton_models/bertQA-ts-script DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/notebooks/conversationalai/model_repo/
```

### Prepare Tacotron 2 and WaveGlow

Now to the final part - TTS system.

Download the [Tacotron 2](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_fp16/files?version=2) and [WaveGlow](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16/files) checkpoints from [NGC](https://ngc.nvidia.com/catalog/models/)
and move them to `Tacotron2/checkpoints/` direcotry:

```bash
mkdir -p DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/checkpoints/
mv nvidia_tacotron2pyt_fp16_20190427 nvidia_waveglow256pyt_fp16 DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/checkpoints/
```

Build the Tacotron 2 container:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/
bash scripts/docker/build.sh
```

Run the container in th interactive mode by typing:
```bash
bash scripts/docker/interactive.sh
```

Export Tacotron 2 to TorchScript:

```bash
cd /workspace/tacotron2/
mkdir -p output
python notebooks/conversationalai/export_tacotron2_ts.py --tacotron2 notebooks/conversationalai/nvidia_tacotron2pyt_fp16.pt -o output/tacotron2_fp16.pt --fp16
```

Export WaveGlow to ONNX intermediate representation:

```bash
python tensorrt/convert_waveglow2onnx.py --waveglow notebooks/conversationalai/nvidia_waveglow256pyt_fp16.pt --wn-channels 256 --fp16 -o output/ --config-file config.json
```

Use the exported ONNX IR to generate TensorRT engine:

```bash
pip install pycuda
python tensorrt/convert_onnx2trt.py --waveglow output/waveglow.onnx -o output/ --fp16
```

After successful export, exit the container and copy the Tacotron 2 model and the WaveGlow engine to `model_repo`:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/
mkdir -p notebooks/conversationalai/model_repo/tacotron2/1/ notebooks/conversationalai/model_repo/waveglow-trt/1/
cp output/tacotron2_fp16.pt notebooks/conversationalai/model_repo/tacotron2/1/
cp output/waveglow_fp16.engine notebooks/conversationalai/model_repo/waveglow-trt/1/
```
## Deployment

Will all models ready for deployment, go to the `conversationalai/client` folder and build the Triron client:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/notebooks/conversationalai/client
docker build -f Dockerfile --network=host -t speech_ai_client:demo .
```

From terminal start the Triton server:

```bash
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/notebooks/conversationalai
NV_GPU=1 nvidia-docker run --ipc=host --network=host --rm -p8000:8000 -p8001:8001 -v ${PWD}/model_repo/:/models nvcr.io/nvidia/tritonserver:20.06-v1-py3 tritonserver --model-store=/models --log-verbose 1

```

In another another terminal run the client:

```bash
docker run -it --rm --network=host --device /dev/snd:/dev/snd speech_ai_client:demo bash /workspace/speech_ai_demo/start_jupyter.sh
```
