# Tacotron 2 and WaveGlow Inference with TensorRT

This is subfolder of the Tacotron 2 for PyTorch repository, tested and maintained by NVIDIA, and provides scripts to perform high-performance inference using NVIDIA TensorRT.

The Tacotron 2 and WaveGlow models form a text-to-speech (TTS) system that enables users to synthesize natural sounding speech from raw transcripts without any additional information such as patterns and/or rhythms of speech. More information about the TTS system and its training can be found in the
[Tacotron 2 PyTorch README](../README.md).

NVIDIA TensorRT is a platform for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high-throughput for deep learning inference applications. After optimizing the compute-intensive acoustic model with NVIDIA TensorRT, inference throughput increased by up to 1.4x over native PyTorch in mixed  precision.


## Quick Start Guide

1. Clone the repository.

	```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
   ```

2. Download pretrained checkpoints from [NGC](https://ngc.nvidia.com/catalog/models) and copy them to the `./checkpoints` directory:

- [Tacotron2 checkpoint](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_fp16)
- [WaveGlow checkpoint](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16)

    ```bash
    mkdir -p checkpoints
    cp <Tacotron2_and_WaveGlow_checkpoints> ./checkpoints/
    ```

3. Build the Tacotron 2 and WaveGlow PyTorch NGC container.

    ```bash
    bash scripts/docker/build.sh
    ```

4. Start an interactive session in the NGC container to run training/inference.
   After you build the container image, you can start an interactive CLI session with:

    ```bash
    bash scripts/docker/interactive.sh
    ```

5. Verify that TensorRT version installed is 7.0 or greater. If necessary, download and install the latest release from https://developer.nvidia.com/nvidia-tensorrt-download

    ```bash
    pip list | grep tensorrt
    dpkg -l | grep TensorRT
    ```

6. Convert the models to ONNX intermediate representation (ONNX IR).
   Convert Tacotron 2 to three ONNX parts: Encoder, Decoder, and Postnet:

	```bash
	mkdir -p output
	python tensorrt/convert_tacotron22onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp16_20190427 -o output/ --fp16
	```

    Convert WaveGlow to ONNX IR:

	```bash
	python tensorrt/convert_waveglow2onnx.py --waveglow ./checkpoints/nvidia_waveglow256pyt_fp16 --config-file config.json --wn-channels 256 -o output/ --fp16
    ```

	After running the above commands, there should be four new ONNX files in `./output/` directory:
    `encoder.onnx`, `decoder_iter.onnx`, `postnet.onnx`, and `waveglow.onnx`.

7. Convert the ONNX IRs to TensorRT engines with fp16 mode enabled:

	```bash
	python tensorrt/convert_onnx2trt.py --encoder output/encoder.onnx --decoder output/decoder_iter.onnx --postnet output/postnet.onnx --waveglow output/waveglow.onnx -o output/ --fp16
	```

	After running the command, there should be four new engine files in `./output/` directory:
    `encoder_fp16.engine`, `decoder_iter_fp16.engine`, `postnet_fp16.engine`, and `waveglow_fp16.engine`.

8. Run TTS inference pipeline with fp16:

	```bash
	python tensorrt/inference_trt.py -i phrases/phrase.txt --encoder output/encoder_fp16.engine --decoder output/decoder_iter_fp16.engine --postnet output/postnet_fp16.engine --waveglow output/waveglow_fp16.engine -o output/ --fp16
	```

## Inference performance: NVIDIA T4

Our results were obtained by running the `./tensorrt/run_latency_tests_trt.sh` script in the PyTorch-19.11-py3 NGC container. Please note that to reproduce the results, you need to provide pretrained checkpoints for Tacotron 2 and WaveGlow. Please edit the script to provide your checkpoint filenames. For all tests in this table, we used WaveGlow with 256 residual channels.

|Framework|Batch size|Input length|Precision|Avg latency (s)|Latency std (s)|Latency confidence interval 90% (s)|Latency confidence interval 95% (s)|Latency confidence interval 99% (s)|Throughput (samples/sec)|Speed-up PyTorch+TensorRT / TensorRT|Avg mels generated (81 mels=1 sec of speech)|Avg audio length (s)|Avg RTF|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PyTorch+TensorRT|1| 128| FP16| 1.02| 0.05| 1.09| 1.10| 1.14| 150,439| 1.59| 602| 6.99| 6.86|
|PyTorch         |1| 128| FP16| 1.63| 0.07| 1.71| 1.73| 1.81|  94,758| 1.00| 601| 6.98| 4.30|
