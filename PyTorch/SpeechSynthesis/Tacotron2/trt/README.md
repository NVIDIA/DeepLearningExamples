# Tacotron 2 and WaveGlow Inference For TensorRT

This is subfolder of the Tacotron 2 for PyTorch repository, tested and
maintained by NVIDIA, and provides scripts to perform high-performance
inference using NVIDIA TensorRT.
The Tacotron 2 and WaveGlow models form a text-to-speech (TTS) system that
enables users to synthesize natural sounding speech from raw transcripts
without any additional information such as patterns and/or rhythms of speech.
More information about the TTS system and its training can be found in the
[Tacotron 2 PyTorch README](../README.md).
NVIDIA TensorRT is a platform for high-performance deep learning inference.
It includes a deep learning inference optimizer and runtime that delivers low
latency and high-throughput for deep learning inference applications. After
optimizing the compute-intensive acoustic model with NVIDIA TensorRT,
inference throughput increased by up to *Xx* over native PyTorch.


## Quick Start Guide

1. Clone the repository.

	```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
    ```

2. Download pretrained checkpoints from [NGC](https://ngc.nvidia.com/catalog/models)
and store them in `./checkpoints` directory:

- [Tacotron2 checkpoint](https://ngc.nvidia.com/models/nvidia:tacotron2pyt_fp16)
- [WaveGlow checkpoint](https://ngc.nvidia.com/models/nvidia:waveglow256pyt_fp16)

    ```bash
    mkdir -p checkpoints
    mv <Tacotron2_checkpoint> <WaveGlow_checkpoint> ./checkpoints/
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

5. Export the models to ONNX intermediate representations (ONNX IRs).
   Export Tacotron 2 to three ONNX parts: Encoder, Decoder, and Postnet:

	```bash
	python exports/export_tacotron2_onnx.py --tacotron2 ./checkpoints/nvidia_tacotron2pyt_fp16_20190427 -o output/
	```

    Export WaveGlow to ONNX IR:

	```bash
	python exports/export_waveglow_onnx.py --waveglow ./checkpoints/nvidia_waveglow256pyt_fp16 --wn-channels 256 -o output/
	```

	After running the above commands, there should be four new files in `./output/`
	directory: `encoder.onnx`, `decoder_iter.onnx`, `postnet.onnx`, and 'waveglow.onnx`.

6. Export the ONNX IRs to TensorRT engines:

	```bash
	python trt/export_onnx2trt.py --encoder output/encoder.onnx --decoder output/decoder_iter.onnx --postnet output/postnet.onnx --waveglow output/waveglow.onnx -o output/ --fp16
	```

	After running the command, there should be four new files in `./output/`
	directory: `encoder_fp16.engine`, `decoder_iter_fp16.engine`, 
	`postnet_fp16.engine`, and 'waveglow_fp16.engine`.

7. Run the inference:

	```bash
	python trt/inference_trt.py -i phrases/phrase.txt --encoder output/encoder_fp16.engine --decoder output/decoder_iter_fp16.engine --postnet output/postnet_fp16.engine --waveglow output/waveglow_fp16.engine -o output/
	```

## Inference performance: NVIDIA T4

Our results were obtained by running the `./trt/run_latency_tests_trt.sh` script in
the PyTorch-19.11-py3 NGC container. Please note that to reproduce the results,
you need to provide pretrained checkpoints for Tacotron 2 and WaveGlow. Please
edit the script to provide your checkpoint filenames. For all tests in this table,
we used WaveGlow with 256 residual channels.

|Framework|Batch size|Input length|Precision|Avg latency (s)|Latency std (s)|Latency confidence interval 90% (s)|Latency confidence interval 95% (s)|Latency confidence interval 99% (s)|Throughput (samples/sec)|Speed-up PyT+TRT/TRT|Avg mels generated (81 mels=1 sec of speech)|Avg audio length (s)|Avg RTF|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PyT+TRT|1| 128| FP16| 1.14| 0.02| 1.16| 1.17| 1.20| 136,865| 1.40| 611| 7.09| 6.20|
|PyT    |1| 128| FP16| 1.58| 0.07| 1.67| 1.70| 1.74| 98,101| 1.00| 605| 7.03| 4.45|
|PyT+TRT|1| 128| FP32| 1.79| 0.01| 1.80| 1.81| 1.84| 86,690| 1.00| 605| 7.02| 3.92|
|PyT    |1| 128| FP32| 1.77| 0.08| 1.88| 1.92| 2.00| 86,529| 1.00| 600| 6.96| 3.92|
