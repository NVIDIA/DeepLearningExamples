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

import sys
import subprocess

import torch


def main():
    argslist = list(sys.argv)[1:]
    world_size = torch.cuda.device_count()

    if '--world-size' in argslist:
        argslist[argslist.index('--world-size') + 1] = str(world_size)
    else:
        argslist.append('--world-size')
        argslist.append(str(world_size))

    workers = []

    for i in range(world_size):
        if '--rank' in argslist:
            argslist[argslist.index('--rank') + 1] = str(i)
        else:
            argslist.append('--rank')
            argslist.append(str(i))
        stdout = None if i == 0 else subprocess.DEVNULL
        worker = subprocess.Popen(
            [str(sys.executable)] + argslist, stdout=stdout)
        workers.append(worker)

    returncode = 0
    try:
        pending = len(workers)
        while pending > 0:
            for worker in workers:
                try:
                    worker_returncode = worker.wait(1)
                except subprocess.TimeoutExpired:
                    continue
                pending -= 1
                if worker_returncode != 0:
                    if returncode != 1:
                        for worker in workers:
                            worker.terminate()
                    returncode = 1

    except KeyboardInterrupt:
        print('Pressed CTRL-C, TERMINATING')
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait()
        raise

    sys.exit(returncode)


if __name__ == "__main__":
    main()
