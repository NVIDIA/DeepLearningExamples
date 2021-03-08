# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

import torch

import tensorrt as trt
from fastspeech.trt import TRT_BASE_PATH, TRT_LOGGER
import fastspeech.trt.common as common
from fastspeech.utils.logging import tprint
from fastspeech.utils.pytorch import to_cpu_numpy, to_gpu_async
from fastspeech.inferencer.waveglow_inferencer import WaveGlowInferencer
from fastspeech.inferencer.denoiser import Denoiser
import pycuda.driver as cuda


class WaveGlowTRTInferencer(object):

    def __init__(self, ckpt_file, engine_file, use_fp16=False, use_denoiser=False, stride=256, n_groups=8):
        self.ckpt_file = ckpt_file
        self.engine_file = engine_file
        self.use_fp16 = use_fp16
        self.use_denoiser = use_denoiser
        self.stride = stride
        self.n_groups = n_groups

        if self.use_denoiser:
            sys.path.append('waveglow')
            waveglow = torch.load(self.ckpt_file)['model']
            waveglow = waveglow.remove_weightnorm(waveglow)
            waveglow.eval()
            self.denoiser = Denoiser(waveglow)
            self.denoiser = to_gpu_async(self.denoiser)
            tprint('Using WaveGlow denoiser.')

            # after initialization, we don't need WaveGlow PyTorch checkpoint
            # anymore - deleting
            del waveglow
            torch.cuda.empty_cache()

        # load engine
        with open(self.engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine:
            tprint('TRT Engine Loaded from {} successfully.'.format(self.engine_file))
            return
        else:
            tprint('Loading TRT Engine from {} failed.'.format(self.engine_file))

    def __enter__(self):
        self.context = self.engine.create_execution_context()

    def __exit__(self, exception_type, exception_value, traceback):
        self.context.__del__()
        self.engine.__del__()

    def infer(self, mels):
        batch_size, _, mel_size = mels.shape
        mels = mels.unsqueeze(3)
        z = torch.randn(batch_size, self.n_groups, mel_size * self.stride // self.n_groups, 1)
        wavs = torch.zeros(batch_size, mel_size * self.stride)

        if self.use_fp16:
            z = z.half()
            mels = mels.half()
            wavs = wavs.half()

        mels = to_gpu_async(mels)
        z = to_gpu_async(z)
        wavs = to_gpu_async(wavs)

        # create inputs/outputs buffers
        input_buffers = common.create_inputs_from_torch(self.engine, [mels, z])
        output_buffers = common.create_outputs_from_torch(self.engine, [wavs.shape])

        # set shapes of inputs
        self.context = common.set_input_shapes(self.engine, self.context, input_buffers)

        # execute
        stream = cuda.Stream()
        bindings = [int(data.data_ptr()) for data in (input_buffers + output_buffers)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()

        wavs = output_buffers[0]

        # denoise
        if self.use_denoiser:
            wavs = self.denoiser(wavs, strength=0.01)

        return wavs.float()