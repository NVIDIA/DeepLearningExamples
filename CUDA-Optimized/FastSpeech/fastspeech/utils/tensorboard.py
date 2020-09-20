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

import matplotlib.pyplot as plt
import numpy as np
import cv2
import data as global_data


plt.switch_backend('Agg')


def image_plot(x, name='image'):
    fig, ax = plt.subplots()
    ax.imshow(x, cmap='magma', aspect='auto')
    fig.canvas.draw()
    buf = np.array(fig.canvas.renderer._renderer)
    plt.clf()
    plt.close('all')
    cv2.imshow(name, buf)
    cv2.waitKey(0)


def plot_to_buf(x, align=True):
    fig, ax = plt.subplots()
    ax.plot(x)
    if align:
        ax.set_ylim([-1, 1])
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer._renderer)
    plt.clf()
    plt.close('all')
    return np.rollaxis(im[..., :3], 2)


def imshow_to_buf(x, scale01=False):
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    if scale01:
        x = (x - x.min()) / (x.max() - x.min())
    if x.max() > 1.:
        x = softmax(x)
    if len(x.shape) == 3:
        x = x[0]
    fig, ax = plt.subplots()
    ax.imshow(x, cmap='magma', aspect='auto')
    fig.canvas.draw()
    im = np.array(fig.canvas.renderer._renderer)
    plt.clf()
    plt.close('all')
    return np.rollaxis(im[..., :3], 2)


def origin_to_chrs(target):
    results = []
    for t in target:
        idx = t - 1 if t - 1 >= 0 else 0
        if idx < len(global_data.idx2chr):
            results.append(global_data.idx2chr[idx])
        else:
            break
    return ''.join(results)