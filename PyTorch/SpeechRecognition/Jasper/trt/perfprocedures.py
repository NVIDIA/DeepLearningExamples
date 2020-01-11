# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''A collection of accuracy and latency evaluation procedures for JASPER on PyTorch and TRT.
'''


import pycuda.driver as cuda
import pycuda.autoinit
import perfutils
import trtutils
import time
import torch
from tqdm import tqdm

def compare_times_trt_pyt_exhaustive(engine, pyt_components, args):
    '''Compares execution times and WER between TRT and PyTorch'''

    # The engine has a fixed-size sequence length, which needs to be known for slicing/padding input
    preprocess_times = []
    inputadjust_times = []
    outputadjust_times = []
    process_batch_times = []
    trt_solo_times = []
    trt_async_times = []
    tohost_sync_times =[]
    pyt_infer_times = []
    step_counter = 0

    with engine.create_execution_context() as context, torch.no_grad():
        for data in tqdm(pyt_components['data_layer'].data_iterator):
            if args.num_steps >= 1:
                if step_counter > args.num_steps:
                    break
                step_counter +=1
            tensors = []
            for d in data:
                tensors.append(d.cuda())
            preprocess_start = time.perf_counter()
            am_input = pyt_components['audio_preprocessor'](tensors[0], tensors[1])
            
            torch.cuda.synchronize()
            preprocess_end = time.perf_counter()

            # Pad or cut to the neccessary engine length
            inputadjust_start = time.perf_counter()
            am_input = perfutils.adjust_shape(am_input, args)
            torch.cuda.synchronize()
            inputadjust_end = time.perf_counter()

            batch_size = am_input[0].shape[0]

            inp = [am_input[0]]
            
            # Run TRT inference 1: Async copying and inference
            # import ipdb; ipdb.set_trace()
            trt_out, time_taken= do_inference_overlap(context, inp)
            torch.cuda.synchronize()
            outputadjust_start = time.perf_counter()
            outputadjust_end = time.perf_counter()
            process_batch_start = time.perf_counter()
            perfutils.global_process_batch(log_probs=trt_out,
                                           original_tensors=tensors,
                                           batch_size=batch_size,
                                           is_trt=True)
            torch.cuda.synchronize()
            process_batch_end = time.perf_counter()

            # Create explicit stream so pytorch doesn't complete asynchronously
            pyt_infer_start = time.perf_counter()
            pyt_out = pyt_components['acoustic_model'](am_input[0])
            torch.cuda.synchronize()
            pyt_infer_end = time.perf_counter()
            perfutils.global_process_batch(log_probs=pyt_out,
                                           original_tensors=tensors,
                                           batch_size=batch_size,
                                           is_trt=False)
            # Run TRT inference 2: Synchronous copying and inference
            sync_out, time_to, time_infer, time_from = do_inference(context,inp)
            del sync_out
            preprocess_times.append(preprocess_end - preprocess_start)
            inputadjust_times.append(inputadjust_end - inputadjust_start)
            outputadjust_times.append(outputadjust_end - outputadjust_start)
            process_batch_times.append(process_batch_end - process_batch_start)
            trt_solo_times.append(time_infer)
            trt_async_times.append(time_taken)
            tohost_sync_times.append(time_from)
            pyt_infer_times.append(pyt_infer_end - pyt_infer_start)

    trt_wer = perfutils.global_process_epoch(is_trt=True)
    pyt_wer = perfutils.global_process_epoch(is_trt=False)
    trt_preds = perfutils._global_trt_dict['predictions']
    pyt_preds = perfutils._global_pyt_dict['predictions']
    times = {
        'preprocess': preprocess_times, # Time to go through preprocessing
        'pyt_infer': pyt_infer_times, # Time for batch completion through pytorch
        'input_adjust': inputadjust_times, # Time to pad/cut for TRT engine size requirements
        'output_adjust' : outputadjust_times, # Time to reshape output of TRT and copy from host to device
        'post_process': process_batch_times, # Time to run greedy decoding and do CTC conversion
        'trt_solo_infer': trt_solo_times, # Time to execute just TRT acoustic model
        'to_host': tohost_sync_times, # Time to execute device to host copy synchronously
        'trt_async_infer': trt_async_times, # Time to execute combined async TRT acoustic model + device to host copy

    }
    wer = {
        'trt': trt_wer,
        'pyt': pyt_wer
    }
    preds = {
        'trt': trt_preds,
        'pyt': pyt_preds
    }
    return wer, preds, times

def do_inference(context, inp):
    '''Do inference using a TRT engine and time it
    Execution and device-to-host copy are completed synchronously
    '''
    # Typical Python-TRT used in samples would copy input data from host to device.
    # Because the PyTorch Tensor is already on the device, such a copy is unneeded.
    t0 = time.perf_counter()
    stream = cuda.Stream()
    # Create output buffers and stream
    outputs, bindings, out_shape = trtutils.allocate_buffers_with_existing_inputs(context, inp)
    t01 = time.perf_counter()
    # simulate sync call here
    context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.handle)
    stream.synchronize()

    t2 = time.perf_counter()
    # for out in outputs:
    #     cuda.memcpy_dtoh(out.host, out.device) 
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
   
    t3 = time.perf_counter()
    copyto = t01-t0
    inference = t2-t01
    copyfrom = t3-t2
    out = outputs[0].host
    outputs[0].device.free()
    out = perfutils.torchify_trt_out(outputs[0].host, out_shape)
    return out, copyto, inference, copyfrom

def do_inference_overlap(context, inp):
    '''Do inference using a TRT engine and time it
    Execution and device-to-host copy are completed asynchronously
    '''
    # Typical Python-TRT used in samples would copy input data from host to device.
    # Because the PyTorch Tensor is already on the device, such a copy is unneeded.
    
    t0 = time.perf_counter()
    # Create output buffers and stream
    stream = cuda.Stream()
    outputs, bindings, out_shape = trtutils.allocate_buffers_with_existing_inputs(context, inp)
    t01 = time.perf_counter()
    t1 = time.perf_counter()
    # Run inference and transfer outputs to host asynchronously
    context.execute_async_v2(
                             bindings=bindings,
                             stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    t2 = time.perf_counter()
    copyto = t1-t0
    inference = t2-t1
    outputs[0].device.free()
    out = perfutils.torchify_trt_out(outputs[0].host, out_shape)
    return out, t2-t1
