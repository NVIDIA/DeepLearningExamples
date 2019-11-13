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

def time_pyt(engine, pyt_components):
    '''Times execution of PyTorch inference
    '''
    baked_seq_len = engine.get_binding_shape(0)[1]
    preprocess_times = []
    pyt_infers = []
    pyt_components['audio_preprocessor'].eval()
    pyt_components['acoustic_model'].eval()
    with torch.no_grad():
        for data in tqdm(pyt_components['data_layer'].data_iterator):
            tensors = []
            for d in data:
                tensors.append(d.to(torch.device("cuda")))
            input_tensor = (tensors[0], tensors[1])
            t0 = time.perf_counter()
            am_input = pyt_components['audio_preprocessor'](x=input_tensor)
            # Pad or cut to the neccessary engine length
            am_input = perfutils.adjust_shape(am_input, baked_seq_len)
            batch_size = am_input[0].shape[0]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # Run PyT inference
            pyt_out = pyt_components['acoustic_model'](x=am_input)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            perfutils.global_process_batch(log_probs=pyt_out,
                                           original_tensors=tensors,
                                           batch_size=batch_size,
                                           is_trt=False)
            assemble_times.append(t1-t0)
            pyt_infers.append(t2-t1)

    pyt_wer = perfutils.global_process_epoch(is_trt=False)
    trt_wer = None
    trt_preds = perfutils._global_trt_dict['predictions']
    pyt_preds = perfutils._global_pyt_dict['predictions']
    times = {
        'preprocess': assemble_times,
        'pyt_infers': pyt_infers
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

def time_trt(engine, pyt_components):
    '''Times execution of TRT inference
    '''
    baked_seq_len = engine.get_binding_shape(0)[1]
    assemble_times = []
    trt_copytos = []
    trt_copyfroms = []
    trt_infers = []
    decodingandeval = []
    with engine.create_execution_context() as context, torch.no_grad():
        for data in tqdm(pyt_components['data_layer'].data_iterator):
            tensors = []
            for d in data:
                tensors.append(d.to(torch.device("cuda")))
            input_tensor = (tensors[0], tensors[1])
            t0 = time.perf_counter()
            am_input = pyt_components['audio_preprocessor'](x=input_tensor)
            # Pad or cut to the neccessary engine length
            am_input = perfutils.adjust_shape(am_input, baked_seq_len)
            batch_size = am_input[0].shape[0]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # Run TRT inference
            trt_out, time_to, time_infer, time_from= do_inference(
                                                                  context=context,
                                                                  inp=am_input,
                                                                  batch_size=batch_size)
            t3 = time.perf_counter()
            trt_out = perfutils.torchify_trt_out(trt_out, batch_size)
            perfutils.global_process_batch(log_probs=trt_out,
                                           original_tensors=tensors,
                                           batch_size=batch_size,
                                           is_trt=True)
            torch.cuda.synchronize()
            t4 = time.perf_counter()


            assemble_times.append(t1-t0)
            trt_copytos.append(time_to)
            trt_copyfroms.append(time_from)
            trt_infers.append(time_infer)
            decodingandeval.append(t4-t3)


    trt_wer = perfutils.global_process_epoch(is_trt=True)
    pyt_wer = perfutils.global_process_epoch(is_trt=False)
    trt_preds = perfutils._global_trt_dict['predictions']
    pyt_preds = perfutils._global_pyt_dict['predictions']
    times = {
        'assemble': assemble_times,
        'trt_copyto': trt_copytos,
        'trt_copyfrom': trt_copyfroms,
        'trt_infers': trt_infers,
        'decodingandeval': decodingandeval
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

def run_trt(engine, pyt_components):
    '''Runs TRT inference for accuracy evaluation
    '''
    baked_seq_len = engine.get_binding_shape(0)[1]
    wers = []
    preds = []
    with engine.create_execution_context() as context, torch.no_grad():
        for data in tqdm(pyt_components['data_layer'].data_iterator):
            tensors = []
            for d in data:
                tensors.append(d.to(torch.device("cuda")))
            input_tensor = (tensors[0], tensors[1])
            am_input = pyt_components['audio_preprocessor'](x=input_tensor)
            # Pad or cut to the neccessary engine length
            am_input = perfutils.adjust_shape(am_input, baked_seq_len)
            batch_size = am_input[0].shape[0]
            torch.cuda.synchronize()
            # Run TRT inference
            trt_out, _,_,_= do_inference(context=context, inp=am_input, batch_size=batch_size)
            trt_out = perfutils.torchify_trt_out(trt_out, batch_size=batch_size)
            wer, pred = perfutils.get_results(log_probs=trt_out,
                                              original_tensors=tensors,
                                              batch_size=batch_size)
            wers.append(wer)
            preds.append(pred)


    return wers, preds

def compare_times_trt_pyt_exhaustive(engine, pyt_components, num_steps):
    '''Compares execution times and WER between TRT and PyTorch'''

    # The engine has a fixed-size sequence length, which needs to be known for slicing/padding input
    baked_seq_len = engine.get_binding_shape(0)[1]
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
            if num_steps >= 1:
                if step_counter > num_steps:
                    break
                step_counter +=1
            tensors = []
            for d in data:
                tensors.append(d.to(torch.device("cuda")))

            input_tensor = (tensors[0], tensors[1])
            preprocess_start = time.perf_counter()
            am_input = pyt_components['audio_preprocessor'](x=input_tensor)
            torch.cuda.synchronize()
            preprocess_end = time.perf_counter()

            # Pad or cut to the neccessary engine length
            inputadjust_start = time.perf_counter()
            am_input = perfutils.adjust_shape(am_input, baked_seq_len)
            torch.cuda.synchronize()
            inputadjust_end = time.perf_counter()

            batch_size = am_input[0].shape[0]

            # Run TRT inference 1: Async copying and inference
            trt_out, time_taken= do_inference_overlap(
                                                      context=context,
                                                      inp=am_input,
                                                      batch_size=batch_size)
            torch.cuda.synchronize()
            outputadjust_start = time.perf_counter()
            trt_out = perfutils.torchify_trt_out(trt_out, batch_size)
            torch.cuda.synchronize()
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
            pyt_out = pyt_components['acoustic_model'](x=am_input[0])
            torch.cuda.synchronize()
            pyt_infer_end = time.perf_counter()
            perfutils.global_process_batch(log_probs=pyt_out,
                                           original_tensors=tensors,
                                           batch_size=batch_size,
                                           is_trt=False)
            # Run TRT inference 2: Synchronous copying and inference
            _, time_to, time_infer, time_from = do_inference(
                                                             context=context,
                                                             inp=am_input,
                                                             batch_size=batch_size)
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

def do_inference(context, inp, batch_size):
    '''Do inference using a TRT engine and time it
    Execution and device-to-host copy are completed synchronously
    '''


    # Typical Python-TRT used in samples would copy input data from host to device.
    # Because the PyTorch Tensor is already on the device, such a copy is unneeded.

    # Create input array of device pointers
    inputs = [inp[0].data_ptr()]
    t0 = time.perf_counter()
    # Create output buffers and stream
    outputs, bindings, stream = trtutils.allocate_buffers_with_existing_inputs(context.engine,
                                                                               inputs,
                                                                               batch_size)
    t1 = time.perf_counter()
    # Run inference and transfer outputs to host asynchronously
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    stream.synchronize()
    t2 = time.perf_counter()
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    t3 = time.perf_counter()


    copyto = t1-t0
    inference = t2-t1
    copyfrom = t3-t2
    out = outputs[0].host
    return out, copyto, inference, copyfrom

def do_inference_overlap(context, inp, batch_size):
    '''Do inference using a TRT engine and time it
    Execution and device-to-host copy are completed asynchronously
    '''
    # Typical Python-TRT used in samples would copy input data from host to device.
    # Because the PyTorch Tensor is already on the device, such a copy is unneeded.

    # Create input array of device pointers
    inputs = [inp[0].data_ptr()]
    t0 = time.perf_counter()
    # Create output buffers and stream
    outputs, bindings, stream = trtutils.allocate_buffers_with_existing_inputs(context.engine,
                                                                               inputs,
                                                                               batch_size)
    t1 = time.perf_counter()
    # Run inference and transfer outputs to host asynchronously
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    t2 = time.perf_counter()


    copyto = t1-t0
    inference = t2-t1
    out = outputs[0].host
    return out, t2-t1
