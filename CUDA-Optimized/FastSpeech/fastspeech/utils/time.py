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

import time
import torch
from fastspeech.utils.logging import tprint

class TimeElapsed(object):
    
    def __init__(self, name, device='cuda', cuda_sync=False, format=""):
        self.name = name
        self.device = device
        self.cuda_sync = cuda_sync
        self.format = format
    
    def __enter__(self):
        self.start()
    
    def __exit__(self, *exc_info):
        self.end()

    def start(self):
        if self.device == 'cuda' and self.cuda_sync:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def end(self):
        if not hasattr(self, "start_time"):
            return
        if self.device == 'cuda' and self.cuda_sync:
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        tprint(("[{}] Time elapsed: {" + self.format + "}").format(self.name, self.time_elapsed))