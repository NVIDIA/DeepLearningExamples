# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import collections
import itertools
import math
import os
import pathlib
import re

import pynvml


class Device:
    # assumes nvml returns list of 64 bit ints
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self):
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_uuid(self):
        return pynvml.nvmlDeviceGetUUID(self.handle)

    def get_cpu_affinity(self):
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string

        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]
        return ret


def get_thread_siblings_list():
    """
    Returns a list of 2-element integer tuples representing pairs of
    hyperthreading cores.
    """
    path = "/sys/devices/system/cpu/cpu*/topology/thread_siblings_list"
    thread_siblings_list = []
    pattern = re.compile(r"(\d+)\D(\d+)")
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(map(int, res[0]))
                thread_siblings_list.append(pair)
    return thread_siblings_list


def check_socket_affinities(socket_affinities):
    # sets of cores should be either identical or disjoint
    for i, j in itertools.product(socket_affinities, socket_affinities):
        if not set(i) == set(j) and not set(i).isdisjoint(set(j)):
            raise RuntimeError(f"Sets of cores should be either identical or disjoint, " f"but got {i} and {j}.")


def get_socket_affinities(nproc_per_node, exclude_unavailable_cores=True):
    devices = [Device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.get_cpu_affinity() for dev in devices]

    if exclude_unavailable_cores:
        available_cores = os.sched_getaffinity(0)
        socket_affinities = [list(set(affinity) & available_cores) for affinity in socket_affinities]

    check_socket_affinities(socket_affinities)

    return socket_affinities


def set_socket_affinity(gpu_id):
    """
    The process is assigned with all available logical CPU cores from the CPU
    socket connected to the GPU with a given id.

    Args:
        gpu_id: index of a GPU
    """
    dev = Device(gpu_id)
    affinity = dev.get_cpu_affinity()
    os.sched_setaffinity(0, affinity)


def set_single_affinity(gpu_id):
    """
    The process is assigned with the first available logical CPU core from the
    list of all CPU cores from the CPU socket connected to the GPU with a given
    id.

    Args:
        gpu_id: index of a GPU
    """
    dev = Device(gpu_id)
    affinity = dev.get_cpu_affinity()

    # exclude unavailable cores
    available_cores = os.sched_getaffinity(0)
    affinity = list(set(affinity) & available_cores)
    os.sched_setaffinity(0, affinity[:1])


def set_single_unique_affinity(gpu_id, nproc_per_node):
    """
    The process is assigned with a single unique available physical CPU core
    from the list of all CPU cores from the CPU socket connected to the GPU with
    a given id.

    Args:
        gpu_id: index of a GPU
    """
    socket_affinities = get_socket_affinities(nproc_per_node)

    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)

    # remove siblings
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(siblings_dict.values()))

    affinities = []
    assigned = []

    for socket_affinity in socket_affinities:
        for core in socket_affinity:
            if core not in assigned:
                affinities.append([core])
                assigned.append(core)
                break
    os.sched_setaffinity(0, affinities[gpu_id])


def set_socket_unique_affinity(gpu_id, nproc_per_node, mode, balanced=True):
    """
    The process is assigned with an unique subset of available physical CPU
    cores from the CPU socket connected to a GPU with a given id.
    Assignment automatically includes hyperthreading siblings (if siblings are
    available).

    Args:
        gpu_id: index of a GPU
        nproc_per_node: total number of processes per node
        mode: mode
        balanced: assign an equal number of physical cores to each process
    """
    socket_affinities = get_socket_affinities(nproc_per_node)

    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)

    # remove hyperthreading siblings
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(siblings_dict.values()))

    socket_affinities_to_device_ids = collections.defaultdict(list)

    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities_to_device_ids[tuple(socket_affinity)].append(idx)

    # compute minimal number of physical cores per GPU across all GPUs and
    # sockets, code assigns this number of cores per GPU if balanced == True
    min_physical_cores_per_gpu = min(
        [len(cores) // len(gpus) for cores, gpus in socket_affinities_to_device_ids.items()]
    )

    for socket_affinity, device_ids in socket_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        if balanced:
            cores_per_device = min_physical_cores_per_gpu
            socket_affinity = socket_affinity[: devices_per_group * min_physical_cores_per_gpu]
        else:
            cores_per_device = len(socket_affinity) // devices_per_group

        for group_id, device_id in enumerate(device_ids):
            if device_id == gpu_id:

                # In theory there should be no difference in performance between
                # 'interleaved' and 'continuous' pattern on Intel-based DGX-1,
                # but 'continuous' should be better for DGX A100 because on AMD
                # Rome 4 consecutive cores are sharing L3 cache.
                # TODO: code doesn't attempt to automatically detect layout of
                # L3 cache, also external environment may already exclude some
                # cores, this code makes no attempt to detect it and to align
                # mapping to multiples of 4.

                if mode == "interleaved":
                    affinity = list(socket_affinity[group_id::devices_per_group])
                elif mode == "continuous":
                    affinity = list(socket_affinity[group_id * cores_per_device: (group_id + 1) * cores_per_device])
                else:
                    raise RuntimeError("Unknown set_socket_unique_affinity mode")

                # unconditionally reintroduce hyperthreading siblings, this step
                # may result in a different numbers of logical cores assigned to
                # each GPU even if balanced == True (if hyperthreading siblings
                # aren't available for a subset of cores due to some external
                # constraints, siblings are re-added unconditionally, in the
                # worst case unavailable logical core will be ignored by
                # os.sched_setaffinity().
                affinity += [siblings_dict[aff] for aff in affinity if aff in siblings_dict]
                os.sched_setaffinity(0, affinity)


def set_affinity(gpu_id, nproc_per_node, mode="socket_unique_continuous", balanced=True):
    """
    The process is assigned with a proper CPU affinity which matches hardware
    architecture on a given platform. Usually it improves and stabilizes
    performance of deep learning training workloads.

    This function assumes that the workload is running in multi-process
    single-device mode (there are multiple training processes and each process
    is running on a single GPU), which is typical for multi-GPU training
    workloads using `torch.nn.parallel.DistributedDataParallel`.

    Available affinity modes:
    * 'socket' - the process is assigned with all available logical CPU cores
    from the CPU socket connected to the GPU with a given id.
    * 'single' - the process is assigned with the first available logical CPU
    core from the list of all CPU cores from the CPU socket connected to the GPU
    with a given id (multiple GPUs could be assigned with the same CPU core).
    * 'single_unique' - the process is assigned with a single unique available
    physical CPU core from the list of all CPU cores from the CPU socket
    connected to the GPU with a given id.
    * 'socket_unique_interleaved' - the process is assigned with an unique
    subset of available physical CPU cores from the CPU socket connected to a
    GPU with a given id, hyperthreading siblings are included automatically,
    cores are assigned with interleaved indexing pattern
    * 'socket_unique_continuous' - (the default) the process is assigned with an
    unique subset of available physical CPU cores from the CPU socket connected
    to a GPU with a given id, hyperthreading siblings are included
    automatically, cores are assigned with continuous indexing pattern

    'socket_unique_continuous' is the recommended mode for deep learning
    training workloads on NVIDIA DGX machines.

    Args:
        gpu_id: integer index of a GPU
        nproc_per_node: number of processes per node
        mode: affinity mode
        balanced: assign an equal number of physical cores to each process,
            affects only 'socket_unique_interleaved' and
            'socket_unique_continuous' affinity modes

    Returns a set of logical CPU cores on which the process is eligible to run.

    Example:

    import argparse
    import os

    import gpu_affinity
    import torch


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--local_rank',
            type=int,
            default=os.getenv('LOCAL_RANK', 0),
        )
        args = parser.parse_args()

        nproc_per_node = torch.cuda.device_count()

        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
        print(f'{args.local_rank}: core affinity: {affinity}')


    if __name__ == "__main__":
        main()

    Launch the example with:
    python -m torch.distributed.launch --nproc_per_node <#GPUs> example.py


    WARNING: On DGX A100 only a half of CPU cores have direct access to GPUs.
    This function restricts execution only to the CPU cores directly connected
    to GPUs, so on DGX A100 it will limit the code to half of CPU cores and half
    of CPU memory bandwidth (which may be fine for many DL models).
    """
    pynvml.nvmlInit()

    if mode == "socket":
        set_socket_affinity(gpu_id)
    elif mode == "single":
        set_single_affinity(gpu_id)
    elif mode == "single_unique":
        set_single_unique_affinity(gpu_id, nproc_per_node)
    elif mode == "socket_unique_interleaved":
        set_socket_unique_affinity(gpu_id, nproc_per_node, "interleaved", balanced)
    elif mode == "socket_unique_continuous":
        set_socket_unique_affinity(gpu_id, nproc_per_node, "continuous", balanced)
    else:
        raise RuntimeError("Unknown affinity mode")

    affinity = os.sched_getaffinity(0)
    return affinity
