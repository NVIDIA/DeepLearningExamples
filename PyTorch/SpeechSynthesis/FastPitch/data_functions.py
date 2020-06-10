# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch

from fastpitch.data_function import (TextMelAliCollate, TextMelAliLoader,
                                      batch_to_gpu as batch_to_gpu_fastpitch)
from tacotron2.data_function import batch_to_gpu as batch_to_gpu_tacotron2
from tacotron2.data_function import TextMelCollate, TextMelLoader
from waveglow.data_function import batch_to_gpu as batch_to_gpu_waveglow
from waveglow.data_function import MelAudioLoader


def get_collate_function(model_name):
    return {'Tacotron2': lambda _: TextMelCollate(n_frames_per_step=1),
            'WaveGlow': lambda _: torch.utils.data.dataloader.default_collate,
            'FastPitch': TextMelAliCollate}[model_name]()

def get_data_loader(model_name, *args):
    return {'Tacotron2': TextMelLoader,
            'WaveGlow': MelAudioLoader,
            'FastPitch': TextMelAliLoader}[model_name](*args)

def get_batch_to_gpu(model_name):
    return {'Tacotron2': batch_to_gpu_tacotron2,
            'WaveGlow': batch_to_gpu_waveglow,
            'FastPitch': batch_to_gpu_fastpitch}[model_name]
