# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math
import os
import pathlib
import re

import pynvml

pynvml.nvmlInit()


def systemGetDriverVersion():
    return pynvml.nvmlSystemGetDriverVersion()


def deviceGetCount():
    return pynvml.nvmlDeviceGetCount()


class device:
    # assume nvml returns list of 64 bit ints
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def getName(self):
        return pynvml.nvmlDeviceGetName(self.handle)

    def getCpuAffinity(self):
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]
        return ret


def set_socket_affinity(gpu_id):
    dev = device(gpu_id)
    affinity = dev.getCpuAffinity()
    os.sched_setaffinity(0, affinity)


def set_single_affinity(gpu_id):
    dev = device(gpu_id)
    affinity = dev.getCpuAffinity()
    os.sched_setaffinity(0, affinity[:1])


def set_single_unique_affinity(gpu_id, world_size):
    devices = [device(i) for i in range(world_size)]
    socket_affinities = [dev.getCpuAffinity() for dev in devices]

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


def set_socket_unique_affinity(gpu_id, world_size, mode):
    device_ids = [device(i) for i in range(world_size)]
    socket_affinities = [dev.getCpuAffinity() for dev in device_ids]

    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)

    # remove siblings
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(siblings_dict.values()))

    socket_affinities_to_device_ids = collections.defaultdict(list)

    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities_to_device_ids[tuple(socket_affinity)].append(idx)

    for socket_affinity, device_ids in socket_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        cores_per_device = len(socket_affinity) // devices_per_group
        for group_id, device_id in enumerate(device_ids):
            if device_id == gpu_id:
                if mode == "interleaved":
                    affinity = list(socket_affinity[group_id::devices_per_group])
                elif mode == "continuous":
                    affinity = list(socket_affinity[group_id * cores_per_device : (group_id + 1) * cores_per_device])
                else:
                    raise RuntimeError("Unknown set_socket_unique_affinity mode")

                # reintroduce siblings
                affinity += [siblings_dict[aff] for aff in affinity if aff in siblings_dict]
                os.sched_setaffinity(0, affinity)


def get_thread_siblings_list():
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


def set_affinity(gpu_id=None, mode="socket"):
    if gpu_id is None:
        gpu_id = int(os.getenv("LOCAL_RANK", 0))

    world_size = int(os.getenv("WORLD_SIZE", 1))

    if mode == "socket":
        set_socket_affinity(gpu_id)
    elif mode == "single":
        set_single_affinity(gpu_id)
    elif mode == "single_unique":
        set_single_unique_affinity(gpu_id, world_size)
    elif mode == "socket_unique_interleaved":
        set_socket_unique_affinity(gpu_id, world_size, "interleaved")
    elif mode == "socket_unique_continuous":
        set_socket_unique_affinity(gpu_id, world_size, "continuous")
    else:
        raise RuntimeError("Unknown affinity mode")

    affinity = os.sched_getaffinity(0)
    return affinity
