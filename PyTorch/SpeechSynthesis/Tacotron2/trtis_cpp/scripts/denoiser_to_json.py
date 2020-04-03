#!/usr/bin/env python3
##
# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     # Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     # Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     # Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
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
# 




import json
import sys
from scipy.signal import get_window
import librosa.util as librosa_util

WAVEGLOW_CONFIG = {
    "n_mel_channels": 80,
    "n_flows": 12,
    "n_group": 8,
    "n_early_every": 4,
    "n_early_size": 2,
    "WN_config": {
        "n_layers": 8,
        "kernel_size": 3,
        "n_channels": 256
    }
}


def gen_win_sq(
        denoiser):
    window = denoiser.stft.window
    win_length = denoiser.stft.win_length
    n_fft = denoiser.stft.filter_length

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=None)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    return win_sq


if len(sys.argv) < 4 or len(sys.argv) > 5:
    print("USAGE:")
    print(
        "\t%s <tacotron2 directory> <waveglow checkpoint> <json output> [strength, default=0.1]" % sys.argv[0])
    sys.exit(1)

json_path = sys.argv[3]

sys.path.append(sys.argv[1])

# must be imported after path is modified
from import_utils import load_waveglow
from waveglow.denoiser import Denoiser

strength = 0.1
if len(sys.argv) == 5:
    strength = float(sys.argv[4])


print("Building denoiser")

waveglow = load_waveglow(sys.argv[2], WAVEGLOW_CONFIG)

denoiser = Denoiser(waveglow).cuda()

statedict = {}

statedict["denoiser.stft.forward_basis"] = denoiser.stft.forward_basis.cpu(
).numpy().tolist()
statedict["denoiser.stft.inverse_basis"] = denoiser.stft.inverse_basis.cpu(
).numpy().tolist()
statedict["denoiser.stft.win_sq"] = gen_win_sq(denoiser).tolist()
statedict["denoiser.bias_spec"] = (
    denoiser.bias_spec*strength).cpu().numpy().tolist()

with open(json_path, "w") as fout:
    json.dump(statedict, fout, indent=2)

print("Wrote to '%s'" % json_path)
