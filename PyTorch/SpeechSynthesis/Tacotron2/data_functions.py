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
from tacotron2.data_function import TextMelCollate
from tacotron2.data_function import TextMelLoader
from waveglow.data_function import MelAudioLoader
from tacotron2.data_function import batch_to_gpu as batch_to_gpu_tacotron2
from waveglow.data_function import batch_to_gpu as batch_to_gpu_waveglow


def get_collate_function(model_name, n_frames_per_step=1):
    if model_name == 'Tacotron2':
        collate_fn = TextMelCollate(n_frames_per_step)
    elif model_name == 'WaveGlow':
        collate_fn = torch.utils.data.dataloader.default_collate
    else:
        raise NotImplementedError(
            "unknown collate function requested: {}".format(model_name))

    return collate_fn


def get_data_loader(model_name, dataset_path, audiopaths_and_text, args):
    if model_name == 'Tacotron2':
        data_loader = TextMelLoader(dataset_path, audiopaths_and_text, args)
    elif model_name == 'WaveGlow':
        data_loader = MelAudioLoader(dataset_path, audiopaths_and_text, args)
    else:
        raise NotImplementedError(
            "unknown data loader requested: {}".format(model_name))

    return data_loader


def get_batch_to_gpu(model_name):
    if model_name == 'Tacotron2':
        batch_to_gpu = batch_to_gpu_tacotron2
    elif model_name == 'WaveGlow':
        batch_to_gpu = batch_to_gpu_waveglow
    else:
        raise NotImplementedError(
            "unknown batch_to_gpu requested: {}".format(model_name))
    return batch_to_gpu
