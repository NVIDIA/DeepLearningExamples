# FastSpeech Inference with TensorRT
This directory contains scripts and usage for accelerated FastSpeech inference with TensorRT. Refer to the [README](../../README.md) for more common and detailed information.

# Parameters
All parameters for TensorRT inference are defined in the default config file, trt.yaml.
```yaml
# File name: fastspeech/hparams/trt.yaml

parent_yaml: 'infer.yaml'

# Inference
batch_size: 1                 # Batch size.
use_trt: True                 # Usage of TensorRT. Must be True to enable TensorRT.
use_fp16: False               # Usage of FP16. Set to True to enable half precision for the engine.

# TRT
trt_file_path: "/workspace/fastspeech/fastspeech.fp16.b1.trt"  # Built TensorRT engine file path.
trt_max_input_seq_len: 128    # Max input sequence length. 
trt_max_output_seq_len: 1024  # Max output sequence length.
trt_max_ws_size: 8            # Max workspace size in GB avaiable for TensorRT engine build.
trt_multi_engine: False       # Usage of multi-engines.
trt_force_build: False        # Force build mode. If True, an engine is forcely built and overwritten to trt_file_path.
```

# Setting TensorRT plugins
Although most of the PyTorch operations can be simply ported to corresponding TensorRT operations, some layers should be manually implemented using tensorRT custom plugins due to the lack of operational supports in the TensorRT Python APIs or for better performance. All plugin code is included in ```fastspeech/trt/plugins```.

## AddPosEnc plugin
AddPosEnc plugin implements "Adding Positional Encoding" layer, which is responsible for computing positional encoding and adding it to the input, i.e., the output is computed as the input plus its positional encoding. This plugin assumes a pair of input and output, with shape of (batch size, time, hidden) and data type of float or half.

```fastspeech/trt/plugins/add_pos_enc``` contains the following.
* ```AddPosEncPlugin.cu```: main CUDA implementation.
* ```AddPosEncPlugin.h```: including plugin meta.
* ```test_add_pos_enc_plugin.py```: specifying how to use the plugin.

Execute "make" and check if ```fastspeech/trt/plugins/add_pos_enc/AddPosEncPlugin.so``` is created sucessfully. It will be loaded during inference.
```
make
```

To test whether the plugin is working correctly, run:
```
python test_add_pos_enc_plugin.py
```

## Repeat plugin
Repeat plugin implements "Tensor Repeat" layer, the main operation in the Length Regulator, e.g., if input=[1,2,3] and repeat=[2,2,3], then output=[1,1,2,2,3,3,3]. This plugin requires two inputs, 1) input sequence, with shape of (batch size, time, hidden) and data type of float, half, or int32, and 2) correspoding repeat count, with shape of (batch size, time) and data type of float(but treated as int internally). The output has the same shape and data type as the input has. According to the maximum output length(```trt_max_output_seq_len```), the plugin output will be properly padded or cropped.

```fastspeech/trt/plugins/repeat```  contains the following.
* ```RepeatPlugin.cu```
* ```RepeatPlugin.h```
* ```test_repeat_plugin.py```

Similarly, execute "make" and check if ```fastspeech/trt/plugins/repeat/RepeatPlugin.so``` is created sucessfully.
```
make
```

To test whether the plugin is working correctly, run:
```
python test_repeat_plugin.py
```

# Inference
In our TensorRT implementation, sequence lengths of input and output are fixed. (Even though TensorRT 7 started to support dynamic shaping, it wasn't used in our implementation because it turned out to not be much effective for the FastSpeech model.)
For the reason, input sequence must be padded or cropped along time dimension to match the max input length (```trt_max_input_seq_len```). And the output will also be padded along the time dimension during inference to match the max output length(```trt_max_output_seq_len```). Make sure that the trained FastSpeech model checkpoint exists in ```checkpoint_path``` before testing inference.

## Building the engine
If any engine file doesn't exist in ```trt_file_path```, the inferencer will first try to build an engine and save it to ```trt_file_path```. The engine build time will take a matter of minutes depending on the settings, i.e., workspace size( ```trt_max_ws_size```), batch size(```batch_size```) and precision(```use_fp16```).

## Using half precision
The inference time could be much improved by using half precision during inference. It may affect the accuracy, but most of the cases, loss of accuracy is neglectable. 

Set --use_fp16 to use half precision. If any engine exists, you can set --trt_force_build to overwrite the engine.

## Using multi-batch
If your system is capable of using multi-batch (either in a real-time setting or batched setting), mutli-batch inference will way more utilize your GPU capability so will give more significant improvement in throughput. 

To enable multi-batch inference, set --batch size. If any engine exists, you can set --trt_force_build to overwrite the engine.

## Using multi-engine (Experimental)
Because engines require fixed shape of input and output through the engines, the input sequence must be padded or cropped to match the max input seqence length(```trt_max_input_seq_len```), regardless of the original length of the sentence. As a result, the paddings could drop the latency and throughput.  To go the extra mile, you can try building multiple engines covering different max input and output seqence lengths and then at run-time, use one of those, which covers the input sequence with least number of paddings.

To enable multi-engine, set --trt_multi_engine, --trt_file_path_list, --trt_max_input_seq_len_list and --trt_max_output_seq_len_list. Refer to the following examples that specifies four different engines' trt_file_path, trt_max_input_seq_len and trt_max_output_seq_len as lists.

```yaml
trt_multi_engine: True
trt_file_path_list: [
  "/fastspeech/preprocessed/fastspeech.i32.o256.trt",
  "/fastspeech/preprocessed/fastspeech.i64.o512.trt",
  "/fastspeech/preprocessed/fastspeech.i96.o768.trt",
  "/fastspeech/preprocessed/fastspeech.i128.o1024.trt",
  ]
trt_max_input_seq_len_list: [32, 64, 96, 128]
trt_max_output_seq_len_list: [256, 512, 768, 1024]
```

Depending on how many engines to use or the GPU memory capacity, it could cause an out-of-memory issue. If any engine exists, you can set --trt_force_build to overwrite the engine.

# Verifying accuracy (Development mode)
Throughout the development of TensorRT inference, you may need to check if the accuracy is kept right. An effective way to verify the accuracy is to compare each activations from every single layers, both from TensorRT and PyTorch inferencers.

```verify_trt.py``` runs inference through both TensorRT and PyTorch models and prints out value differences for every activations. Since the engine additionaly outputs intermediate activations besides final outputs, if any engine already exists, you must set --trt_force_build to overwrite the engine.

* --```text```: a input text to feed-forward. (optional)
```
python fastspeech/trt/verify_trt.py --hparam=trt.yaml --text="It just works." --trt_force_build
```

The diff of activations and outputs are printed as the following format.
```
...

# out.seq #  (activation or output name)

[PyTorch]
tensor([[[-6.7603, -6.4677, -6.1418,  ..., -9.7428, -9.6195, -9.5378],
         [-6.4849, -5.9863, -5.4004,  ..., -7.2682, -6.9044, -6.7582],
         [-6.2242, -5.6636, -5.1177,  ..., -6.3784, -5.8876, -5.6169],
         ...,
         [-7.0700, -6.6304, -6.3764,  ..., -8.7196, -8.4675, -8.3227],
         [-7.2344, -6.8167, -6.5161,  ..., -9.4241, -9.3277, -9.2811],
         [-6.2658, -6.3746, -6.3305,  ..., -9.5452, -9.4297, -9.3122]]],
       device='cuda:0')

[TRT]:
tensor([[[-6.7603, -6.4677, -6.1418,  ..., -9.7428, -9.6195, -9.5378],
         [-6.4849, -5.9863, -5.4004,  ..., -7.2682, -6.9044, -6.7582],
         [-6.2242, -5.6636, -5.1177,  ..., -6.3784, -5.8876, -5.6169],
         ...,
         [-7.0700, -6.6304, -6.3764,  ..., -8.7196, -8.4675, -8.3227],
         [-7.2344, -6.8167, -6.5161,  ..., -9.4241, -9.3277, -9.2811],
         [-6.2658, -6.3746, -6.3305,  ..., -9.5452, -9.4297, -9.3122]]],
       device='cuda:0')

[Diff]:
tensor([ 4.7684e-07, -9.5367e-07, -9.5367e-07,  ..., -9.5367e-07,
        -9.5367e-07, -9.5367e-07], device='cuda:0')

[Errors]:
tensor([ 4.7684e-07, -9.5367e-07, -9.5367e-07,  ..., -9.5367e-07,
        -9.5367e-07, -9.5367e-07], device='cuda:0')
- identical? False
- 8160 errors out of 10240
- max: 7.62939453125e-06
```

# Performance
## NVIDIA V100
| Framework | Batch size | Precision | Latency(s) | Speed-up (PyT - PyT+TRT)
|---------|----|------|----------|------|
| PyT     | 1  | FP16 | 0.024345 | 1    |
| PyT+TRT | 1  | FP16 | 0.007161 | 3.03 |

## NVIDIA T4
| Framework | Batch size | Precision | Latency(s) | Speed-up (PyT - PyT+TRT)
|---------|----|------|----------|------|
| PyT     | 1  | FP16 | 0.026932 | 1    |
| PyT+TRT | 1  | FP16 | 0.014594 | 1.79 |
