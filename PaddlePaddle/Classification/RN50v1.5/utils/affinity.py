# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

import os
import logging
import paddle


def _get_gpu_affinity_table():
    """
    Generate three dict objects, gpu_cpu_affinity_map, cpu_socket_gpus_list, cpu_core_groups.
    gpu_cpu_affinity_map (dict): Key is GPU ID and value is cpu_affinity string.
    cpu_socket_gpus_list (dict): Key is cpu_affinity string and value is  a list
                                 collected all GPU IDs that affinity to this cpu socket.
    cpu_core_groups (dict):      Key is cpu_affinity string and value is cpu core groups.
                                 cpu core groups contains #GPUs groups, each group have,
                                 nearly eaual amount of cpu cores.

    Example:
        $nvidis-smi topo -m
            GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
        GPU0     X     SYS     SYS     SYS      0-9,20-29           0
        GPU1   SYS       X     SYS     SYS      0-9,20-29           0
        GPU2   SYS      SYS      X     SYS      10-19,30-39         1
        GPU3   SYS      SYS    SYS       X      10-19,30-39         1

        gpu_cpu_affinity_map =
            { 0: '0-9,20-29', # GPU0's cpu affninity is '0-9,20-29'
              1: '0-9,20-29', # GPU1's cpu affninity is '0-9,20-29'
              2: '10-19,30-39', # GPU2's cpu affninity is '10-19,30-39'
              3: '10-19,30-39' } # GPU3's cpu affninity is '10-19,30-39'
        cpu_socket_gpus_list =
            { '0-9,20-29': [0, 1], # There are 2 GPUs, 0 and 1, belong to cpu affinity '0-9,20-29'.
              '10-19,30-39': [2, 3] # There are 2 GPUs, 2 and 3, belong to cpu affinity '10-19,30-39'.
            }
        cpu_core_groups =
            # There are 2 GPUs belong to cpu affinity '0-9,20-29', then
            # cores [0, 1, ..., 8, 9] would be split to two groups every
            # 2-th elements
            # [0, 2, 4, 6, 8] and [1, 3, 5, 7, 9]
            # The same for cores [20, 21, ..., 28, 29].
            {'0-9,20-29': [
                               [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
                               [[20, 22, 24, 26, 28], [21, 23, 25, 27, 29]]
                              ],
            # The same as '0-9,20-29'
            '10-19,30-39': [
                            [[10, 12, 14, 16, 18], [11, 13, 15, 17, 19]],
                            [[30, 32, 34, 36, 38], [31, 33, 35, 37, 39]]
                           ]}

    """
    lines = os.popen('nvidia-smi topo -m').readlines()

    cpu_affinity_idx = -1
    titles = lines[0].split('\t')
    for idx in range(len(titles)):
        if 'CPU Affinity' in titles[idx]:
            cpu_affinity_idx = idx
    assert cpu_affinity_idx > 0, \
        "Can not obtain correct CPU affinity column index via nvidia-smi!"

    gpu_cpu_affinity_map = dict()
    cpu_socket_gpus_list = dict()
    # Skip title
    for idx in range(1, len(lines)):
        line = lines[idx]
        items = line.split('\t')

        if 'GPU' in items[0]:
            gpu_id = int(items[0][3:])
            affinity = items[cpu_affinity_idx]
            gpu_cpu_affinity_map[gpu_id] = affinity
            if affinity in cpu_socket_gpus_list:
                cpu_socket_gpus_list[affinity].append(gpu_id)
            else:
                cpu_socket_gpus_list[affinity] = [gpu_id]

    cpu_core_groups = _group_cpu_cores(cpu_socket_gpus_list)
    return gpu_cpu_affinity_map, cpu_socket_gpus_list, cpu_core_groups


def _group_cpu_cores(cpu_socket_gpus_list):
    """
    Generate a dictionary that key is cpu_affinity string and value is cpu core groups.
    cpu core groups contains #GPUs groups, each group have, nearly eaual amount of cpu cores.
    The grouping way is collect cpu cores every #GPUs-th elements, due to index of hyperthreading.
    For examle, 4 physical cores, 8 cores with hyperthreading. The CPU indices [0, 1, 2, 3] is
    physical cores, and [4, 5, 6, 7] is hyperthreading. In this case, distributing physical cores
    first, then hyperthreading would reach better performance.
    Args:
        cpu_socket_gpus_list (dict): a dict that map cpu_affinity_str to all GPUs that belong to it.
    Return:
        cpu_core_groups (dict): a dict that map cpu_affinity_str to cpu core groups.
    Example:
        cpu_socket_gpus_list = { '0-9,20-29': [0, 1], '10-19,30-39': [2, 3] },
        which means there are 2 GPUs, 0 and 1, belong to '0-9,20-29' and
        2 GPUs, 2 and 3, belong to '10-19,30-39'
        therefore, cpu_core_groups =
                {'0-9,20-29': [
                               [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
                               [[20, 22, 24, 26, 28], [21, 23, 25, 27, 29]]
                              ],
                 '10-19,30-39': [
                                 [[10, 12, 14, 16, 18], [11, 13, 15, 17, 19]],
                                 [[30, 32, 34, 36, 38], [31, 33, 35, 37, 39]]
                                ]}

    """
    cpu_core_groups = dict()
    for cpu_socket in cpu_socket_gpus_list:
        cpu_core_groups[cpu_socket] = list()
        gpu_count = len(cpu_socket_gpus_list[cpu_socket])
        cores = cpu_socket.split(',')
        for core in cores:
            core_indices = _get_core_indices(core)
            core_group = list()
            for i in range(gpu_count):
                start = i % len(core_indices)
                sub_core_set = core_indices[start::gpu_count]
                core_group.append(sub_core_set)
            cpu_core_groups[cpu_socket].append(core_group)
    return cpu_core_groups


def _get_core_indices(cores_str):
    """
    Generate a dictionary of cpu core indices.
    Args:
        cores_str (str): a string with format "start_idx-end_idx".
    Return:
        cpu_core_indices (list): a list collected all indices in [start_idx, end_idx].
    Example:
        cores_str = '0-20'
        cpu_core_indices = [0, 1, 2, ..., 18, 19, 20]
    """
    start, end = cores_str.split('-')
    return [*range(int(start), int(end) + 1)]


def set_cpu_affinity():
    """
    Setup CPU affinity.
    Each GPU would be bound to a specific set of CPU cores for optimal and stable performance.
    This function would obtain GPU-CPU affinity via "nvidia-smi topo -m", then equally distribute
    CPU cores to each GPU.
    """

    gpu_cpu_affinity_map, cpu_socket_gpus_list, cpu_core_groups = \
        _get_gpu_affinity_table()

    node_num = paddle.distributed.fleet.node_num()
    gpu_per_node = paddle.distributed.get_world_size() // node_num
    local_rank = paddle.distributed.get_rank() % gpu_per_node

    # gpu_cpu_affinity_map (dict): Key is GPU ID and value is cpu_affinity string.
    # cpu_socket_gpus_list (dict): Key is cpu_affinity string and value is  a list
    #                              collected all GPU IDs that affinity to this cpu socket.
    # cpu_core_groups (dict):      Key is cpu_affinity string and value is cpu core groups.
    #                              cpu core groups contains #GPUs groups, each group have,
    #                              nearly eaual amount of cpu cores.
    # Example:
    # $nvidis-smi topo -m
    #        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
    # GPU0     X     SYS     SYS     SYS      0-9,20-29           0
    # GPU1   SYS       X     SYS     SYS      0-9,20-29           0
    # GPU2   SYS      SYS      X     SYS      10-19,30-39         1
    # GPU3   SYS      SYS    SYS       X      10-19,30-39         1
    #
    # gpu_cpu_affinity_map =
    #     { 0: '0-9,20-29',
    #       1: '0-9,20-29',
    #       2: '10-19,30-39',
    #       3: '10-19,30-39' }
    # cpu_socket_gpus_list =
    #     { '0-9,20-29': [0, 1],
    #       '10-19,30-39': [2, 3] }
    # cpu_core_groups =
    #     {'0-9,20-29': [
    #                     [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
    #                     [[20, 22, 24, 26, 28], [21, 23, 25, 27, 29]]
    #                    ],
    #       '10-19,30-39': [
    #                        [[10, 12, 14, 16, 18], [11, 13, 15, 17, 19]],
    #                        [[30, 32, 34, 36, 38], [31, 33, 35, 37, 39]]
    #                       ]}
    #
    # for rank-0, it belong to '0-9,20-29' cpu_affinity_key,
    # and it locate in index-0 of cpu_socket_gpus_list['0-9,20-29'],
    # therefore, affinity_mask would be a collection of all cpu cores
    # in index-0 of cpu_core_groups['0-9,20-29'], that is [0, 2, 4, 6, 8]
    # and [20, 22, 24, 26, 28].
    # affinity_mask = [0, 2, 4, 6, 8, 20, 22, 24, 26, 28]
    affinity_mask = list()
    cpu_affinity_key = gpu_cpu_affinity_map[local_rank]
    cpu_core_idx = cpu_socket_gpus_list[cpu_affinity_key].index(local_rank)
    for cpu_core_group in cpu_core_groups[cpu_affinity_key]:
        affinity_mask.extend(cpu_core_group[cpu_core_idx])

    pid = os.getpid()
    os.sched_setaffinity(pid, affinity_mask)
    logging.info("Set CPU affinity of rank-%d (Process %d) "
                 "to %s.", local_rank, pid, str(os.sched_getaffinity(pid)))
